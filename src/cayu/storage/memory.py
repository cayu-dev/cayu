from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections import Counter
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from math import sqrt
from typing import Any, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
MAX_KNOWLEDGE_CHUNK_INDEX = 2**31 - 1
MAX_KNOWLEDGE_REVISION = 2**31 - 1
_SEARCH_TOKEN_RE = re.compile(r"\w+")
_SHA256_HEX_RE = re.compile(r"[0-9a-f]{64}\Z")

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
    entry_id: str
    content_hash: str
    model: str
    dimensions: int | None
    vector: list[float]


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


class KnowledgeRevisionConflict(RuntimeError):
    """A canonical write lost a compare-and-swap race."""

    def __init__(
        self,
        entry_id: str,
        *,
        expected_revision: int | None,
        actual_revision: int | None,
    ) -> None:
        self.entry_id = require_clean_nonblank(entry_id, "entry_id")
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

    @field_validator("id", "namespace", "kind", "created_by")
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

    @field_validator("id", "entry_id")
    @classmethod
    def validate_clean_nonblank_fields(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

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
        if _knowledge_query_has_positive_terms(self):
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

    @field_validator("query")
    @classmethod
    def copy_query(cls, value):
        return copy_knowledge_query(value)

    @field_validator("hits")
    @classmethod
    def copy_hits(cls, value):
        return [copy_knowledge_hit(hit) for hit in value]

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
    entry_id: str = Field(max_length=256)
    entry_revision: int
    expected_revision: int | None
    request_sha256: str
    entry_created_at: datetime
    entry_updated_at: datetime
    committed_at: datetime
    replayed: bool = False

    @field_validator("operation_id", "entry_id")
    @classmethod
    def validate_clean_ids(cls, value: str, info) -> str:
        value = require_clean_nonblank(value, info.field_name)
        if len(value.encode("utf-8")) > 256:
            raise ValueError(f"`{info.field_name}` must be at most 256 UTF-8 bytes.")
        return value

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
        access_scope: KnowledgeAccessScope | None = None,
        operation_id: str,
        expected_revision: int | None = None,
    ) -> KnowledgePublicationReceipt:
        """Publish one create/append exactly once with immutable replay evidence.

        Implementations commit the revision, chunks, current pointer, and receipt
        atomically. ``expected_revision=None`` creates revision 1; a positive
        value appends exactly its successor.
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
    ) -> None:
        self._default_access_scope = (
            None if access_scope is None else copy_knowledge_access_scope(access_scope)
        )
        self._entries: dict[str, dict[int, KnowledgeEntry]] = {}
        self._current_revisions: dict[str, int] = {}
        self._chunks: dict[tuple[str, int], list[KnowledgeChunk]] = {}
        self._publication_receipts: dict[str, KnowledgePublicationReceipt] = {}
        self._publication_access: dict[str, _KnowledgeAccessSnapshot] = {}
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

    async def create_entry(
        self,
        entry: KnowledgeEntry,
        chunks: list[KnowledgeChunk] | None = None,
        *,
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
        self._require_chunk_ids_available(
            copied_chunks,
            access_scope=scope,
            operation="create_entry",
        )
        self._entries[entry.id] = {1: entry}
        self._current_revisions[entry.id] = 1
        self._chunks[(entry.id, 1)] = copied_chunks
        return copy_knowledge_entry(entry)

    async def append_entry_revision(
        self,
        entry: KnowledgeEntry,
        chunks: list[KnowledgeChunk] | None = None,
        *,
        expected_revision: int,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeEntry:
        scope = self._operation_access_scope(access_scope)
        entry = copy_knowledge_entry(entry)
        _validate_revision_append(entry, expected_revision=expected_revision)
        current = self._current_entry(entry.id)
        if current is None:
            raise KnowledgeRevisionConflict(
                entry.id,
                expected_revision=expected_revision,
                actual_revision=None,
            )
        _require_knowledge_entry_access(scope, current, operation="append_entry_revision")
        if current.revision != expected_revision:
            raise KnowledgeRevisionConflict(
                entry.id,
                expected_revision=expected_revision,
                actual_revision=current.revision,
            )
        _validate_revision_successor(current, entry)
        _require_knowledge_successor_access(
            scope,
            entry,
            operation="append_entry_revision",
        )
        copied_chunks = self._revision_chunks(entry, chunks, previous=current)
        self._require_chunk_ids_available(
            copied_chunks,
            access_scope=scope,
            operation="append_entry_revision",
        )
        self._entries[entry.id][entry.revision] = entry
        self._chunks[(entry.id, entry.revision)] = copied_chunks
        self._current_revisions[entry.id] = entry.revision
        return copy_knowledge_entry(entry)

    async def get_entry(
        self,
        entry_id: str,
        *,
        revision: int | None = None,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeEntry | None:
        scope = self._operation_access_scope(access_scope)
        clean_id = require_clean_nonblank(entry_id, "entry_id")
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
        clean_id = require_clean_nonblank(entry_id, "entry_id")
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
        return await self.append_entry_revision(
            updated,
            expected_revision=expected_revision,
            access_scope=scope,
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
        clean_id = require_clean_nonblank(entry_id, "entry_id")
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
            self._entries.pop(clean_id, None)
            self._current_revisions.pop(clean_id, None)
            for key in [key for key in self._chunks if key[0] == clean_id]:
                self._chunks.pop(key, None)
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
        cutoff = datetime.now(UTC) if now is None else now
        expired_ids = [
            entry_id
            for entry_id in self._entries
            if (entry := self._current_entry(entry_id)) is not None
            if entry.expires_at is not None
            and entry.expires_at <= cutoff
            and _knowledge_scope_allows_entry(scope, entry, now=cutoff)
        ]
        for entry_id in expired_ids:
            self._entries.pop(entry_id, None)
            self._current_revisions.pop(entry_id, None)
            for key in [key for key in self._chunks if key[0] == entry_id]:
                self._chunks.pop(key, None)
        return len(expired_ids)

    async def publish_entry_revision(
        self,
        entry: KnowledgeEntry,
        chunks: list[KnowledgeChunk],
        *,
        access_scope: KnowledgeAccessScope | None = None,
        operation_id: str,
        expected_revision: int | None = None,
    ) -> KnowledgePublicationReceipt:
        scope = self._operation_access_scope(access_scope)
        operation_id, copied_entry, copied_chunks, request_sha256 = prepare_knowledge_publication(
            entry,
            chunks,
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
        self._entries.setdefault(copied_entry.id, {})[copied_entry.revision] = copied_entry
        self._chunks[(copied_entry.id, copied_entry.revision)] = copied_chunks
        self._current_revisions[copied_entry.id] = copied_entry.revision
        self._publication_receipts[operation_id] = receipt
        self._publication_access[operation_id] = _knowledge_access_snapshot(copied_entry)
        return copy_knowledge_publication_receipt(receipt)

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
        clean_id = require_clean_nonblank(entry_id, "entry_id")
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
        embedding_dimensions: int | None = None,
        entries: list[KnowledgeEntry] | None = None,
        access_scope: KnowledgeAccessScope | None = None,
        hybrid_keyword_weight: float = 0.35,
        semantic_min_score: float = 0.55,
    ) -> None:
        if not isinstance(embedding_provider, TextEmbeddingProvider):
            raise TypeError("embedding_provider must implement TextEmbeddingProvider.")
        self.embedding_provider = embedding_provider
        self.embedding_model = require_clean_nonblank(embedding_model, "embedding_model")
        if embedding_dimensions is not None:
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
        super().__init__(entries, access_scope=access_scope)

    def supported_search_modes(self) -> tuple[KnowledgeSearchMode, ...]:
        return (
            KnowledgeSearchMode.AUTO,
            KnowledgeSearchMode.KEYWORD,
            KnowledgeSearchMode.SEMANTIC,
            KnowledgeSearchMode.HYBRID,
        )

    async def create_entry(
        self,
        entry: KnowledgeEntry,
        chunks: list[KnowledgeChunk] | None = None,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeEntry:
        stored = await super().create_entry(
            entry,
            access_scope=access_scope,
            chunks=chunks,
        )
        await self._embed_entry_chunks(stored.id, stored.revision)
        return stored

    async def append_entry_revision(
        self,
        entry: KnowledgeEntry,
        chunks: list[KnowledgeChunk] | None = None,
        *,
        expected_revision: int,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeEntry:
        stored = await super().append_entry_revision(
            entry,
            expected_revision=expected_revision,
            access_scope=access_scope,
            chunks=chunks,
        )
        await self._embed_entry_chunks(stored.id, stored.revision)
        self._drop_stale_entry_embeddings(stored.id)
        return stored

    async def delete_entry(
        self,
        entry_id: str,
        *,
        expected_revision: int,
        access_scope: KnowledgeAccessScope | None = None,
        hard: bool = False,
    ) -> KnowledgeEntry | None:
        deleted = await super().delete_entry(
            entry_id,
            expected_revision=expected_revision,
            access_scope=access_scope,
            hard=hard,
        )
        if hard and deleted is not None:
            self._drop_entry_embeddings(deleted.id)
        return deleted

    async def prune_expired(
        self,
        *,
        access_scope: KnowledgeAccessScope | None = None,
        now: datetime | None = None,
    ) -> int:
        scope = self._operation_access_scope(access_scope)
        cutoff = datetime.now(UTC) if now is None else now
        expired_ids = [
            entry_id
            for entry_id in self._entries
            if (entry := self._current_entry(entry_id)) is not None
            if entry.expires_at is not None
            and entry.expires_at <= cutoff
            and _knowledge_scope_allows_entry(scope, entry, now=cutoff)
        ]
        pruned = await super().prune_expired(access_scope=scope, now=cutoff)
        for entry_id in expired_ids:
            self._drop_entry_embeddings(entry_id)
        return pruned

    async def publish_entry_revision(
        self,
        entry: KnowledgeEntry,
        chunks: list[KnowledgeChunk],
        *,
        access_scope: KnowledgeAccessScope | None = None,
        operation_id: str,
        expected_revision: int | None = None,
    ) -> KnowledgePublicationReceipt:
        receipt = await super().publish_entry_revision(
            entry,
            chunks,
            access_scope=access_scope,
            operation_id=operation_id,
            expected_revision=expected_revision,
        )
        if receipt.replayed:
            return receipt
        stored_chunks = self._chunks.get((receipt.entry_id, receipt.entry_revision), [])
        await self._embed_chunks(stored_chunks)
        self._drop_stale_entry_embeddings(receipt.entry_id)
        return receipt

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
        if not candidates:
            return KnowledgeSearchResult(
                query=knowledge_query,
                hits=[],
                truncated=False,
                limit=knowledge_query.limit,
                max_bytes=knowledge_query.max_bytes,
                total_hits_known=0,
            )
        semantic_query_text = _semantic_query_text(knowledge_query)
        candidate_embeddings = await self._embed_chunks(
            [chunk for _, chunks in candidates for chunk in chunks]
        )
        query_vector = await self._embed_query(knowledge_query, semantic_query_text)
        semantic_min_score = (
            self.semantic_min_score
            if knowledge_query.min_score is None
            else knowledge_query.min_score
        )
        scored: list[
            tuple[float, KnowledgeEntry, KnowledgeChunk | None, str, str, float | None, bool]
        ] = []
        for entry, chunks in candidates:
            semantic_score, chunk = self._best_semantic_score(
                chunks,
                candidate_embeddings,
                query_vector,
            )
            if semantic_score is None:
                continue
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
        return _search_result_from_scored_embeddings(scored, knowledge_query, score_kind=score_kind)

    async def _embed_entry_chunks(self, entry_id: str, revision: int) -> None:
        await self._embed_chunks(self._chunks.get((entry_id, revision), []))

    async def _embed_chunks(self, chunks: list[KnowledgeChunk]) -> dict[str, list[float]]:
        vectors = {
            chunk.id: list(self._chunk_embeddings[chunk.id]["vector"])
            for chunk in chunks
            if self._has_current_embedding(chunk)
        }
        missing = [chunk for chunk in chunks if not self._has_current_embedding(chunk)]
        if not missing:
            return vectors
        result = copy_text_embedding_result(
            await self.embedding_provider.embed_texts(
                TextEmbeddingRequest(
                    model=self.embedding_model,
                    texts=[chunk.text for chunk in missing],
                    dimensions=self.embedding_dimensions,
                )
            )
        )
        if len(result.embeddings) != len(missing):
            raise ValueError("Embedding provider returned a different number of embeddings.")
        by_index = {embedding.index: embedding for embedding in result.embeddings}
        for index, chunk in enumerate(missing):
            embedding = by_index.get(index)
            if embedding is None:
                raise ValueError("Embedding provider did not return every requested index.")
            self._validate_embedding_dimension(embedding.vector)
            vector = list(embedding.vector)
            vectors[chunk.id] = vector
            if not self._chunk_is_current(chunk):
                continue
            self._chunk_embeddings[chunk.id] = {
                "entry_id": chunk.entry_id,
                "content_hash": _knowledge_chunk_content_hash(chunk),
                "model": self.embedding_model,
                "dimensions": self.embedding_dimensions,
                "vector": vector,
            }
        return vectors

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
        embedding = next((item for item in result.embeddings if item.index == 0), None)
        if embedding is None:
            raise ValueError("Embedding provider did not return query embedding index 0.")
        self._validate_embedding_dimension(embedding.vector)
        return list(embedding.vector)

    def _validate_embedding_dimension(self, vector: list[float]) -> None:
        if self.embedding_dimensions is not None and len(vector) != self.embedding_dimensions:
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

    def _has_current_embedding(self, chunk: KnowledgeChunk) -> bool:
        stored = self._chunk_embeddings.get(chunk.id)
        return (
            stored is not None
            and stored["entry_id"] == chunk.entry_id
            and stored["content_hash"] == _knowledge_chunk_content_hash(chunk)
            and stored["model"] == self.embedding_model
            and stored["dimensions"] == self.embedding_dimensions
        )

    def _drop_entry_embeddings(self, entry_id: str) -> None:
        stale_ids = [
            chunk_id
            for chunk_id, embedding in self._chunk_embeddings.items()
            if embedding["entry_id"] == entry_id
        ]
        for chunk_id in stale_ids:
            self._chunk_embeddings.pop(chunk_id, None)

    def _drop_stale_entry_embeddings(
        self,
        entry_id: str,
    ) -> None:
        current = self._current_entry(entry_id)
        current_ids = (
            set()
            if current is None
            else {chunk.id for chunk in self._chunks.get((entry_id, current.revision), [])}
        )
        stale_ids = [
            chunk_id
            for chunk_id, embedding in self._chunk_embeddings.items()
            if embedding["entry_id"] == entry_id and chunk_id not in current_ids
        ]
        for chunk_id in stale_ids:
            self._chunk_embeddings.pop(chunk_id, None)

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
    if snapshot.status not in scope.allowed_statuses:
        return False
    cutoff = datetime.now(UTC) if now is None else now
    return scope.include_expired or snapshot.expires_at is None or snapshot.expires_at > cutoff


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
    operation_id: str,
    expected_revision: int | None = None,
) -> tuple[str, KnowledgeEntry, list[KnowledgeChunk], str]:
    """Copy and bind one complete revision-publication authority tuple."""

    clean_operation_id = _knowledge_publication_operation_id(operation_id)
    copied_entry = copy_knowledge_entry(entry)
    _validate_revision_append(copied_entry, expected_revision=expected_revision)
    copied_chunks = _copy_entry_chunks(
        copied_entry.id,
        copied_entry.revision,
        chunks,
    )
    request_sha256 = sha256(
        canonical_durable_json_bytes(
            {
                "contract": "cayu-knowledge-revision-publication-v1",
                "expected_revision": expected_revision,
                "entry": copied_entry.model_dump(mode="json"),
                "chunks": [chunk.model_dump(mode="json") for chunk in copied_chunks],
            },
            "knowledge publication",
        )
    ).hexdigest()
    return clean_operation_id, copied_entry, copied_chunks, request_sha256


def _knowledge_publication_operation_id(operation_id: str) -> str:
    clean = require_clean_nonblank(operation_id, "operation_id")
    if len(clean.encode("utf-8")) > 256:
        raise ValueError("`operation_id` must be at most 256 UTF-8 bytes.")
    return clean


def _validate_knowledge_publication_replay(
    receipt: KnowledgePublicationReceipt,
    *,
    entry: KnowledgeEntry,
    request_sha256: str,
) -> None:
    receipt = copy_knowledge_publication_receipt(receipt)
    if (
        receipt.entry_id != entry.id
        or receipt.entry_revision != entry.revision
        or receipt.request_sha256 != request_sha256
        or receipt.entry_created_at != entry.created_at
        or receipt.entry_updated_at != entry.updated_at
    ):
        raise KnowledgePublicationConflict("operation_mismatch")


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
    if chunk.content_hash is not None:
        return chunk.content_hash
    return f"sha256:{sha256(chunk.text.encode('utf-8')).hexdigest()}"


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


def _knowledge_query_has_positive_terms(query: KnowledgeQuery) -> bool:
    if _tokenize_search_text(query.text or ""):
        return True
    return bool(query.any_terms or query.all_terms or query.phrases)


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


def _validate_knowledge_revision(value: int, field_name: str) -> None:
    _validate_positive_int(value, field_name)
    if value > MAX_KNOWLEDGE_REVISION:
        raise ValueError(f"`{field_name}` must be at most {MAX_KNOWLEDGE_REVISION}.")


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
