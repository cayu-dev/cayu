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
    copy_json_value,
    copy_label_map,
    require_clean_nonblank,
    require_finite,
    require_nonblank,
)
from cayu.embeddings import TextEmbeddingProvider, TextEmbeddingRequest

DEFAULT_KNOWLEDGE_NAMESPACE = "default"
DEFAULT_KNOWLEDGE_KIND = "fact"
DEFAULT_KNOWLEDGE_LIMIT = 10
DEFAULT_KNOWLEDGE_MAX_BYTES = 20_000
_SEARCH_TOKEN_RE = re.compile(r"\w+")

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
    phrases: list[str]


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


class KnowledgeEntry(BaseModel):
    """Durable, source-attributed knowledge item."""

    model_config = ConfigDict(extra="forbid")

    id: str
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
        return copy_json_value(value, "metadata")

    @field_validator("labels", mode="before")
    @classmethod
    def copy_labels(cls, value) -> dict[str, str]:
        return copy_label_map(value, "labels")

    @field_validator("id", "namespace", "kind", "created_by")
    @classmethod
    def validate_clean_nonblank_fields(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

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
        copied = copy_json_value(value, info.field_name)
        if type(copied) is not list:
            raise ValueError(f"`{info.field_name}` must be a list.")
        result: list[str] = []
        for index, item in enumerate(copied):
            if type(item) is not str:
                raise ValueError(f"`{info.field_name}[{index}]` must be a string.")
            result.append(require_clean_nonblank(item, f"{info.field_name}[{index}]"))
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
    model_config = ConfigDict(extra="forbid")

    id: str
    entry_id: str
    text: str
    chunk_index: int
    content_hash: str | None = None
    source_uri: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "metadata")

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
        return value


class KnowledgeQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    model_config = ConfigDict(extra="forbid")

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

    @model_validator(mode="after")
    def validate_chunk_belongs_to_entry(self) -> KnowledgeHit:
        if self.chunk is not None and self.chunk.entry_id != self.entry.id:
            raise ValueError("`chunk.entry_id` must match `entry.id`.")
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


class KnowledgeStore(ABC):
    """Searchable knowledge contract."""

    def supported_search_modes(self) -> tuple[KnowledgeSearchMode, ...]:
        """Return search modes this store can execute directly."""

        return (KnowledgeSearchMode.AUTO, KnowledgeSearchMode.KEYWORD)

    @abstractmethod
    async def put_entry(self, entry: KnowledgeEntry) -> KnowledgeEntry:
        """Insert or update one knowledge entry by id."""

    @abstractmethod
    async def get_entry(self, entry_id: str) -> KnowledgeEntry | None:
        """Load one entry by id."""

    @abstractmethod
    async def update_entry_status(
        self,
        entry_id: str,
        status: KnowledgeStatus,
    ) -> KnowledgeEntry:
        """Update one entry status and return the updated entry."""

    @abstractmethod
    async def delete_entry(
        self,
        entry_id: str,
        *,
        hard: bool = False,
    ) -> KnowledgeEntry | None:
        """Soft-delete one entry by default, or hard-delete when requested."""

    @abstractmethod
    async def replace_chunks(
        self, entry_id: str, chunks: list[KnowledgeChunk]
    ) -> list[KnowledgeChunk]:
        """Replace the complete chunk set for an existing entry."""

    @abstractmethod
    async def put_entry_with_chunks(
        self,
        entry: KnowledgeEntry,
        chunks: list[KnowledgeChunk],
    ) -> KnowledgeEntry:
        """Atomically write one entry and its complete chunk set."""

    @abstractmethod
    async def read_chunks(
        self,
        entry_id: str,
        *,
        chunk_index: int | None = None,
        around: int = 0,
        max_chunks: int = DEFAULT_KNOWLEDGE_LIMIT,
        max_bytes: int = DEFAULT_KNOWLEDGE_MAX_BYTES,
    ) -> list[KnowledgeChunk]:
        """Read bounded chunks for one entry."""

    @abstractmethod
    async def search(self, query: KnowledgeQuery) -> KnowledgeSearchResult:
        """Search knowledge and return a bounded result envelope."""

    @abstractmethod
    async def list_entries(self, query: KnowledgeListQuery) -> KnowledgeListResult:
        """List entries/facets for discovery without requiring a lexical search term."""


class InMemoryKnowledgeStore(KnowledgeStore):
    """In-memory knowledge store for tests, demos, and single-process apps."""

    def __init__(self, entries: list[KnowledgeEntry] | None = None) -> None:
        self._entries: dict[str, KnowledgeEntry] = {}
        self._chunks: dict[str, list[KnowledgeChunk]] = {}
        if entries:
            for entry in entries:
                copied = copy_knowledge_entry(entry)
                if copied.id in self._entries:
                    raise ValueError(f"Duplicate knowledge entry id {copied.id!r}.")
                self._entries[copied.id] = copied
                self._chunks[copied.id] = [_default_chunk_for_entry(copied)]

    async def put_entry(self, entry: KnowledgeEntry) -> KnowledgeEntry:
        entry = copy_knowledge_entry(entry)
        existing_entry = self._entries.get(entry.id)
        existing_chunks = self._chunks.get(entry.id)
        self._entries[entry.id] = entry
        if (
            existing_entry is None
            or existing_chunks is None
            or _has_only_default_chunk(existing_entry, existing_chunks)
        ):
            self._chunks[entry.id] = [_default_chunk_for_entry(entry)]
        return copy_knowledge_entry(entry)

    async def get_entry(self, entry_id: str) -> KnowledgeEntry | None:
        clean_id = require_clean_nonblank(entry_id, "entry_id")
        entry = self._entries.get(clean_id)
        if entry is None:
            return None
        return copy_knowledge_entry(entry)

    async def update_entry_status(
        self,
        entry_id: str,
        status: KnowledgeStatus,
    ) -> KnowledgeEntry:
        clean_id = require_clean_nonblank(entry_id, "entry_id")
        entry = self._entries.get(clean_id)
        if entry is None:
            raise KeyError(f"Knowledge entry {clean_id!r} does not exist.")
        updated = entry.model_copy(update={"status": status, "updated_at": _next_updated_at(entry)})
        updated = copy_knowledge_entry(updated)
        self._entries[clean_id] = updated
        return copy_knowledge_entry(updated)

    async def delete_entry(
        self,
        entry_id: str,
        *,
        hard: bool = False,
    ) -> KnowledgeEntry | None:
        clean_id = require_clean_nonblank(entry_id, "entry_id")
        entry = self._entries.get(clean_id)
        if entry is None:
            return None
        if hard:
            self._entries.pop(clean_id, None)
            self._chunks.pop(clean_id, None)
            return copy_knowledge_entry(entry)
        return await self.update_entry_status(clean_id, KnowledgeStatus.DELETED)

    async def replace_chunks(
        self, entry_id: str, chunks: list[KnowledgeChunk]
    ) -> list[KnowledgeChunk]:
        clean_id = require_clean_nonblank(entry_id, "entry_id")
        if clean_id not in self._entries:
            raise KeyError(f"Knowledge entry {clean_id!r} does not exist.")
        copied_chunks = _copy_entry_chunks(clean_id, chunks)
        self._chunks[clean_id] = copied_chunks
        return [copy_knowledge_chunk(chunk) for chunk in copied_chunks]

    async def put_entry_with_chunks(
        self,
        entry: KnowledgeEntry,
        chunks: list[KnowledgeChunk],
    ) -> KnowledgeEntry:
        copied_entry = copy_knowledge_entry(entry)
        copied_chunks = _copy_entry_chunks(copied_entry.id, chunks)
        self._entries[copied_entry.id] = copied_entry
        self._chunks[copied_entry.id] = copied_chunks
        return copy_knowledge_entry(copied_entry)

    async def read_chunks(
        self,
        entry_id: str,
        *,
        chunk_index: int | None = None,
        around: int = 0,
        max_chunks: int = DEFAULT_KNOWLEDGE_LIMIT,
        max_bytes: int = DEFAULT_KNOWLEDGE_MAX_BYTES,
    ) -> list[KnowledgeChunk]:
        clean_id = require_clean_nonblank(entry_id, "entry_id")
        if clean_id not in self._entries:
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
        chunks = self._chunks.get(clean_id, [])
        if chunk_index is not None:
            chunks = _center_chunk_window(chunks, chunk_index=chunk_index, max_chunks=max_chunks)
        return _bounded_chunks(
            chunks,
            start_index=start_index,
            end_index=end_index,
            max_chunks=max_chunks,
            max_bytes=max_bytes,
        )

    async def search(self, query: KnowledgeQuery) -> KnowledgeSearchResult:
        knowledge_query = copy_knowledge_query(query)
        if knowledge_query.mode not in {KnowledgeSearchMode.AUTO, KnowledgeSearchMode.KEYWORD}:
            raise ValueError("InMemoryKnowledgeStore supports only auto and keyword search modes.")
        scored: list[tuple[float, KnowledgeEntry, KnowledgeChunk | None, str, str]] = []
        for entry in self._entries.values():
            if not _entry_matches_query(entry, knowledge_query):
                continue
            score, chunk, reason, preview_text = _score_entry(
                entry, self._chunks.get(entry.id, []), knowledge_query
            )
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
            if len(preview.encode("utf-8")) < source_bytes:
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

    async def list_entries(self, query: KnowledgeListQuery) -> KnowledgeListResult:
        knowledge_query = copy_knowledge_list_query(query)
        entries = [
            entry
            for entry in self._entries.values()
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
            if len(preview.encode("utf-8")) < len(preview_source.encode("utf-8")):
                truncated = True
            remaining -= len(preview.encode("utf-8"))
            items.append(
                KnowledgeListItem(
                    entry=entry,
                    chunk_count=len(self._chunks.get(entry.id, [])),
                    text_preview=preview,
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
        super().__init__(entries)

    def supported_search_modes(self) -> tuple[KnowledgeSearchMode, ...]:
        return (
            KnowledgeSearchMode.AUTO,
            KnowledgeSearchMode.KEYWORD,
            KnowledgeSearchMode.SEMANTIC,
            KnowledgeSearchMode.HYBRID,
        )

    async def put_entry(self, entry: KnowledgeEntry) -> KnowledgeEntry:
        stored = await super().put_entry(entry)
        await self._embed_entry_chunks(stored.id)
        return stored

    async def delete_entry(
        self,
        entry_id: str,
        *,
        hard: bool = False,
    ) -> KnowledgeEntry | None:
        deleted = await super().delete_entry(entry_id, hard=hard)
        if hard and deleted is not None:
            self._drop_entry_embeddings(deleted.id)
        return deleted

    async def replace_chunks(
        self,
        entry_id: str,
        chunks: list[KnowledgeChunk],
    ) -> list[KnowledgeChunk]:
        stored_chunks = await super().replace_chunks(entry_id, chunks)
        await self._embed_chunks(stored_chunks)
        self._drop_stale_entry_embeddings(entry_id, stored_chunks)
        return stored_chunks

    async def put_entry_with_chunks(
        self,
        entry: KnowledgeEntry,
        chunks: list[KnowledgeChunk],
    ) -> KnowledgeEntry:
        stored = await super().put_entry_with_chunks(entry, chunks)
        stored_chunks = self._chunks.get(stored.id, [])
        await self._embed_chunks(stored_chunks)
        self._drop_stale_entry_embeddings(stored.id, stored_chunks)
        return stored

    async def search(self, query: KnowledgeQuery) -> KnowledgeSearchResult:
        knowledge_query = copy_knowledge_query(query)
        if knowledge_query.mode is KnowledgeSearchMode.KEYWORD:
            return await super().search(knowledge_query)
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
        candidates = [
            entry
            for entry in self._entries.values()
            if _entry_matches_query(entry, knowledge_query)
            and not _entry_matches_none_terms(entry, self._chunks.get(entry.id, []), terms)
        ]
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
        await self._embed_entries(candidates)
        query_vector = await self._embed_query(knowledge_query, semantic_query_text)
        scored: list[
            tuple[float, KnowledgeEntry, KnowledgeChunk | None, str, str, float | None]
        ] = []
        for entry in candidates:
            semantic_score, chunk = self._best_semantic_score(entry, query_vector)
            if semantic_score is None:
                continue
            normalized_semantic = _normalize_cosine_similarity(semantic_score)
            semantic_matched = normalized_semantic >= self.semantic_min_score
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
                    self._chunks.get(entry.id, []),
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
            scored.append((score, entry, chunk, reason, preview_text, score_normalized))
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

    async def _embed_entries(self, entries: list[KnowledgeEntry]) -> None:
        chunks: list[KnowledgeChunk] = []
        for entry in entries:
            chunks.extend(self._chunks.get(entry.id, []))
        await self._embed_chunks(chunks)

    async def _embed_entry_chunks(self, entry_id: str) -> None:
        await self._embed_chunks(self._chunks.get(entry_id, []))

    async def _embed_chunks(self, chunks: list[KnowledgeChunk]) -> None:
        missing = [chunk for chunk in chunks if not self._has_current_embedding(chunk)]
        if not missing:
            return
        result = await self.embedding_provider.embed_texts(
            TextEmbeddingRequest(
                model=self.embedding_model,
                texts=[chunk.text for chunk in missing],
                dimensions=self.embedding_dimensions,
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
            self._chunk_embeddings[chunk.id] = {
                "entry_id": chunk.entry_id,
                "content_hash": _knowledge_chunk_content_hash(chunk),
                "model": self.embedding_model,
                "dimensions": self.embedding_dimensions,
                "vector": list(embedding.vector),
            }

    async def _embed_query(self, query: KnowledgeQuery, text: str) -> list[float]:
        result = await self.embedding_provider.embed_texts(
            TextEmbeddingRequest(
                model=self.embedding_model,
                texts=[text],
                dimensions=self.embedding_dimensions,
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
        entry: KnowledgeEntry,
        query_vector: list[float],
    ) -> tuple[float | None, KnowledgeChunk | None]:
        best_score: float | None = None
        best_chunk: KnowledgeChunk | None = None
        for chunk in self._chunks.get(entry.id, []):
            stored = self._chunk_embeddings.get(chunk.id)
            if stored is None or stored["content_hash"] != _knowledge_chunk_content_hash(chunk):
                continue
            score = _cosine_similarity(query_vector, stored["vector"])
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
        chunks: list[KnowledgeChunk],
    ) -> None:
        current_ids = {chunk.id for chunk in chunks}
        stale_ids = [
            chunk_id
            for chunk_id, embedding in self._chunk_embeddings.items()
            if embedding["entry_id"] == entry_id and chunk_id not in current_ids
        ]
        for chunk_id in stale_ids:
            self._chunk_embeddings.pop(chunk_id, None)


def copy_knowledge_entry(entry: KnowledgeEntry) -> KnowledgeEntry:
    if type(entry) is not KnowledgeEntry:
        raise TypeError("KnowledgeEntry instances must not be subclasses.")
    return KnowledgeEntry(
        id=entry.id,
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
        metadata=copy_json_value(entry.metadata, "metadata"),
    )


def copy_knowledge_chunk(chunk: KnowledgeChunk) -> KnowledgeChunk:
    if type(chunk) is not KnowledgeChunk:
        raise TypeError("KnowledgeChunk instances must not be subclasses.")
    return KnowledgeChunk(
        id=chunk.id,
        entry_id=chunk.entry_id,
        text=chunk.text,
        chunk_index=chunk.chunk_index,
        content_hash=chunk.content_hash,
        source_uri=chunk.source_uri,
        metadata=copy_json_value(chunk.metadata, "metadata"),
    )


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
    )


def copy_knowledge_list_item(item: KnowledgeListItem) -> KnowledgeListItem:
    if type(item) is not KnowledgeListItem:
        raise TypeError("KnowledgeListItem instances must not be subclasses.")
    return KnowledgeListItem(
        entry=copy_knowledge_entry(item.entry),
        chunk_count=item.chunk_count,
        text_preview=item.text_preview,
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


def _copy_entry_chunks(entry_id: str, chunks: list[KnowledgeChunk]) -> list[KnowledgeChunk]:
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
        if chunk.id in seen_ids:
            raise ValueError("Knowledge chunk ids must be unique within an entry.")
        if chunk.chunk_index in seen_indexes:
            raise ValueError("Knowledge chunk indexes must be unique within an entry.")
        seen_ids.add(chunk.id)
        seen_indexes.add(chunk.chunk_index)
    return sorted(copied_chunks, key=lambda chunk: chunk.chunk_index)


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
        chunk_search_text = _entry_chunk_searchable_text(entry, chunk)
        chunk_score = _score_candidate(chunk_search_text, terms)
        if chunk_score > best_score:
            best_score = chunk_score
            best_chunk = chunk
            best_reason = "chunk text match"
            best_preview_text = chunk.text
    return best_score, best_chunk, best_reason, best_preview_text


def _score_candidate(text: str, terms: _SearchTerms) -> float:
    if not _text_matches_structured_terms(text, terms):
        return 0.0
    token_counts = Counter(_tokenize_search_text(text))
    score = float(sum(token_counts[term] for term in terms["any"]))
    score += float(sum(max(token_counts[term] for term in group) for group in terms["all"]))
    folded = text.casefold()
    score += float(sum(2 for phrase in terms["phrases"] if phrase in folded))
    return score


def _text_matches_structured_terms(text: str, terms: _SearchTerms) -> bool:
    tokens = set(_tokenize_search_text(text))
    folded = text.casefold()
    if any(term in tokens for term in terms["none"]):
        return False
    if not all(any(term in tokens for term in group) for group in terms["all"]):
        return False
    positives = terms["any"] or terms["phrases"]
    return not positives or (
        any(term in tokens for term in terms["any"])
        or any(phrase in folded for phrase in terms["phrases"])
    )


def _entry_chunk_searchable_text(entry: KnowledgeEntry, chunk: KnowledgeChunk) -> str:
    parts: list[str] = []
    if entry.title is not None:
        parts.append(entry.title)
    parts.append(entry.text)
    if chunk.text == entry.text:
        return "\n".join(parts)
    parts.append(chunk.text)
    return "\n".join(parts)


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
    scored: list[tuple[float, KnowledgeEntry, KnowledgeChunk | None, str, str, float | None]],
    query: KnowledgeQuery,
    *,
    score_kind: str,
) -> KnowledgeSearchResult:
    hits: list[KnowledgeHit] = []
    remaining = query.max_bytes
    truncated = False
    for rank, (score, entry, chunk, reason, preview_text, normalized_score) in enumerate(
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
        if len(preview.encode("utf-8")) < source_bytes:
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
        "phrases": _dedupe_strings([_normalize_search_phrase(phrase) for phrase in query.phrases]),
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


def _normalize_search_phrase(value: str) -> str:
    return require_nonblank(value, "phrase").casefold()


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
        id=f"{entry.id}:0",
        entry_id=entry.id,
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
