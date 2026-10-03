"""Backend-independent knowledge search and listing contracts."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cayu._validation import copy_json_value, copy_label_map, require_finite
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu._validation import require_durable_nonblank as require_nonblank
from cayu.knowledge.indexing import KnowledgeIndexCoverage, copy_knowledge_index_coverage
from cayu.knowledge.records import (
    DEFAULT_KNOWLEDGE_LIMIT,
    DEFAULT_KNOWLEDGE_MAX_BYTES,
    DEFAULT_KNOWLEDGE_NAMESPACE,
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeStatus,
    KnowledgeVisibility,
    _dedupe_strings,
    _validate_nonnegative_int,
    _validate_positive_int,
    copy_knowledge_chunk,
    copy_knowledge_entry,
)

MAX_KNOWLEDGE_QUERY_ASPECT_GROUPS = 6


MAX_KNOWLEDGE_QUERY_ASPECTS_PER_GROUP = 128


MAX_KNOWLEDGE_QUERY_GROUPED_ASPECTS = (
    MAX_KNOWLEDGE_QUERY_ASPECT_GROUPS * MAX_KNOWLEDGE_QUERY_ASPECTS_PER_GROUP
)


MAX_KNOWLEDGE_QUERY_GROUPED_ASPECT_BYTES = 128_000


_SEARCH_TOKEN_RE = re.compile(r"\w+")


class _SearchTerms(TypedDict):
    any: list[str]
    all: list[list[str]]
    none: list[str]
    phrases: list[list[str]]


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
    aspect_groups: list[list[str]] = Field(default_factory=list)
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

    @field_validator("aspect_groups", mode="before")
    @classmethod
    def copy_aspect_groups(cls, value) -> list[list[str]]:
        if value is None:
            return []
        copied = copy_json_value(value, "aspect_groups")
        if type(copied) is not list:
            raise ValueError("`aspect_groups` must be a list.")
        if len(copied) > MAX_KNOWLEDGE_QUERY_ASPECT_GROUPS:
            raise ValueError(
                "`aspect_groups` cannot contain more than "
                f"{MAX_KNOWLEDGE_QUERY_ASPECT_GROUPS} groups."
            )
        groups: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()
        total_values = 0
        total_bytes = 0
        for group_index, group in enumerate(copied):
            if type(group) is not list or not group:
                raise ValueError(f"`aspect_groups[{group_index}]` must be a non-empty list.")
            if len(group) > MAX_KNOWLEDGE_QUERY_ASPECTS_PER_GROUP:
                raise ValueError(
                    f"`aspect_groups[{group_index}]` cannot contain more than "
                    f"{MAX_KNOWLEDGE_QUERY_ASPECTS_PER_GROUP} values."
                )
            values: list[str] = []
            for value_index, item in enumerate(group):
                if type(item) is not str:
                    raise ValueError(
                        f"`aspect_groups[{group_index}][{value_index}]` must be a string."
                    )
                normalized_item = require_clean_nonblank(
                    item,
                    f"aspect_groups[{group_index}][{value_index}]",
                )
                total_bytes += len(normalized_item.encode("utf-8"))
                values.append(normalized_item)
            normalized = tuple(_dedupe_strings(values))
            if normalized in seen:
                raise ValueError("`aspect_groups` cannot contain duplicate groups.")
            seen.add(normalized)
            groups.append(list(normalized))
            total_values += len(normalized)
        if total_values > MAX_KNOWLEDGE_QUERY_GROUPED_ASPECTS:
            raise ValueError(
                "`aspect_groups` cannot contain more than "
                f"{MAX_KNOWLEDGE_QUERY_GROUPED_ASPECTS} distinct values."
            )
        if total_bytes > MAX_KNOWLEDGE_QUERY_GROUPED_ASPECT_BYTES:
            raise ValueError(
                "`aspect_groups` cannot exceed "
                f"{MAX_KNOWLEDGE_QUERY_GROUPED_ASPECT_BYTES} UTF-8 bytes."
            )
        return groups

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
        has_raw_semantic_text = self.mode is KnowledgeSearchMode.SEMANTIC and self.text is not None
        if (
            _query_terms_have_positive_terms(terms)
            or has_raw_semantic_text
            or (
                self.aspect_groups
                and self.mode in {KnowledgeSearchMode.AUTO, KnowledgeSearchMode.KEYWORD}
            )
        ):
            return self
        raise ValueError(
            "Knowledge query requires positive search terms, raw semantic text, "
            "or exact aspect groups for auto/keyword search."
        )


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
        aspect_groups=[list(group) for group in query.aspect_groups],
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
