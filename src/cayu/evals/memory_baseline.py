from __future__ import annotations

import json
from datetime import datetime
from math import ceil, log2
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cayu._validation import (
    canonical_durable_json_bytes,
    copy_durable_json_object,
    copy_label_map,
    require_durable_clean_nonblank,
    require_durable_nonblank,
)
from cayu.storage import (
    KnowledgeAccessScope,
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeListQuery,
    KnowledgeQuery,
    KnowledgeSearchMode,
    KnowledgeStatus,
    KnowledgeStore,
    KnowledgeVisibility,
)

MEMORY_RETRIEVAL_CORPUS_SCHEMA_VERSION = "cayu.memory_retrieval_corpus.v1"
MEMORY_RETRIEVAL_BASELINE_SCHEMA_VERSION = "cayu.memory_retrieval_baseline.v1"
_MAX_CORPUS_BYTES = 4 * 1024 * 1024
_MAX_CORPUS_ENTRIES = 10_000
_MAX_CORPUS_CASES = 10_000


class MemoryRetrievalAccessSpec(BaseModel):
    """Application-derived access fixture; it intentionally contains no tenant model."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    allowed_namespaces: tuple[str, ...]
    required_labels: dict[str, str] = Field(default_factory=dict)
    allowed_visibilities: tuple[KnowledgeVisibility, ...] = (KnowledgeVisibility.GLOBAL,)
    allowed_source_types: tuple[str, ...] | None = None
    allowed_source_ids: tuple[str, ...] | None = None
    allowed_statuses: tuple[KnowledgeStatus, ...] = (KnowledgeStatus.ACTIVE,)
    include_expired: bool = False

    @field_validator("required_labels", mode="before")
    @classmethod
    def copy_required_labels(cls, value) -> dict[str, str]:
        return copy_label_map(value, "required_labels")

    @model_validator(mode="after")
    def validate_nonempty_boundaries(self) -> MemoryRetrievalAccessSpec:
        if not self.allowed_namespaces:
            raise ValueError("allowed_namespaces must not be empty.")
        if not self.allowed_visibilities:
            raise ValueError("allowed_visibilities must not be empty.")
        if not self.allowed_statuses:
            raise ValueError("allowed_statuses must not be empty.")
        return self

    def to_scope(self) -> KnowledgeAccessScope:
        return KnowledgeAccessScope(
            allowed_namespaces=list(self.allowed_namespaces),
            required_labels=dict(self.required_labels),
            allowed_visibilities=list(self.allowed_visibilities),
            allowed_source_types=(
                None if self.allowed_source_types is None else list(self.allowed_source_types)
            ),
            allowed_source_ids=(
                None if self.allowed_source_ids is None else list(self.allowed_source_ids)
            ),
            allowed_statuses=list(self.allowed_statuses),
            include_expired=self.include_expired,
        )


class MemoryRetrievalCorpusEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    id: str
    text: str
    namespace: str
    labels: dict[str, str] = Field(default_factory=dict)
    visibility: KnowledgeVisibility = KnowledgeVisibility.GLOBAL
    status: KnowledgeStatus = KnowledgeStatus.ACTIVE
    source_type: str | None = None
    source_id: str | None = None
    language: str
    expires_at: datetime | None = None

    @field_validator("id", "namespace", "language")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return require_durable_nonblank(value, "text")

    @field_validator("source_type", "source_id")
    @classmethod
    def validate_optional_identity(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("labels", mode="before")
    @classmethod
    def copy_labels(cls, value) -> dict[str, str]:
        return copy_label_map(value, "labels")

    def to_entry(self) -> KnowledgeEntry:
        return KnowledgeEntry(
            id=self.id,
            text=self.text,
            namespace=self.namespace,
            labels=dict(self.labels),
            visibility=self.visibility,
            status=self.status,
            source_type=self.source_type,
            source_id=self.source_id,
            expires_at=self.expires_at,
        )


class MemoryRetrievalCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    id: str
    trajectory_id: str
    turn_index: int
    language: str
    query: str
    namespace: str
    access: MemoryRetrievalAccessSpec
    relevant_entry_ids: tuple[str, ...]
    forbidden_entry_ids: tuple[str, ...] = ()
    expected_source_ids: dict[str, str] = Field(default_factory=dict)
    cutoff: int = 5

    @field_validator("id", "trajectory_id", "language", "namespace")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        return require_durable_nonblank(value, "query")

    @field_validator("turn_index", mode="before")
    @classmethod
    def validate_turn_index(cls, value) -> int:
        if type(value) is not int or value < 0:
            raise ValueError("turn_index must be a non-negative integer.")
        return value

    @field_validator("cutoff", mode="before")
    @classmethod
    def validate_cutoff(cls, value) -> int:
        if type(value) is not int or value < 1:
            raise ValueError("cutoff must be a positive integer.")
        return value

    @model_validator(mode="after")
    def validate_relevance(self) -> MemoryRetrievalCase:
        if not self.relevant_entry_ids:
            raise ValueError("relevant_entry_ids must not be empty.")
        if len(self.relevant_entry_ids) != len(set(self.relevant_entry_ids)):
            raise ValueError("relevant_entry_ids must be unique.")
        if set(self.relevant_entry_ids) & set(self.forbidden_entry_ids):
            raise ValueError("Relevant and forbidden entry ids must not overlap.")
        if set(self.expected_source_ids) != set(self.relevant_entry_ids):
            raise ValueError("expected_source_ids must cover every relevant entry exactly.")
        return self


class MemoryRetrievalIdProbe(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    id: str
    access: MemoryRetrievalAccessSpec
    forbidden_entry_ids: tuple[str, ...]

    @model_validator(mode="after")
    def validate_probe(self) -> MemoryRetrievalIdProbe:
        require_durable_clean_nonblank(self.id, "id")
        if not self.forbidden_entry_ids:
            raise ValueError("forbidden_entry_ids must not be empty.")
        return self


class MemoryRetrievalCorpus(BaseModel):
    """Portable public or private corpus accepted by the same hermetic runner."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal["cayu.memory_retrieval_corpus.v1"]
    corpus_revision: str
    origin: Literal["hermetic_public", "external_private"]
    entries: tuple[MemoryRetrievalCorpusEntry, ...]
    cases: tuple[MemoryRetrievalCase, ...]
    id_probes: tuple[MemoryRetrievalIdProbe, ...] = ()

    @field_validator("corpus_revision")
    @classmethod
    def validate_revision(cls, value: str) -> str:
        return require_durable_clean_nonblank(value, "corpus_revision")

    @model_validator(mode="after")
    def validate_references(self) -> MemoryRetrievalCorpus:
        if not self.entries or len(self.entries) > _MAX_CORPUS_ENTRIES:
            raise ValueError(f"entries must contain 1..{_MAX_CORPUS_ENTRIES} items.")
        if not self.cases or len(self.cases) > _MAX_CORPUS_CASES:
            raise ValueError(f"cases must contain 1..{_MAX_CORPUS_CASES} items.")
        entry_ids = [entry.id for entry in self.entries]
        if len(entry_ids) != len(set(entry_ids)):
            raise ValueError("Corpus entry ids must be unique.")
        case_ids = [case.id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Corpus case ids must be unique.")
        known = set(entry_ids)
        referenced = {
            entry_id
            for case in self.cases
            for entry_id in (*case.relevant_entry_ids, *case.forbidden_entry_ids)
        } | {entry_id for probe in self.id_probes for entry_id in probe.forbidden_entry_ids}
        unknown = sorted(referenced - known)
        if unknown:
            raise ValueError(f"Corpus cases reference unknown entries: {unknown!r}.")
        return self


class MemoryRetrievalCaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    case_id: str
    language: str
    selected_entry_ids: tuple[str, ...]
    candidate_count: int
    truncated: bool
    recall_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float
    false_injection_count: int
    stale_result_count: int
    authorization_leak_count: int
    citation_correct_count: int
    citation_evaluated_count: int
    model_facing_bytes: int
    latency_ms: float


class MemoryRetrievalLanguageSlice(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    language: str
    case_count: int
    recall_at_k: float
    mean_reciprocal_rank: float
    mean_ndcg_at_k: float


class MemoryRetrievalBaselineMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    case_count: int
    recall_at_k: float
    mean_reciprocal_rank: float
    mean_ndcg_at_k: float
    false_injection_rate: float
    stale_result_rate: float
    authorization_leak_rate: float
    citation_correctness: float
    latency_p50_ms: float
    latency_p95_ms: float
    total_candidate_count: int
    mean_candidate_count: float
    truncated_case_count: int
    model_facing_bytes: int
    estimated_model_facing_tokens: int


class MemoryRetrievalBaselineResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal["cayu.memory_retrieval_baseline.v1"]
    corpus_revision: str
    corpus_origin: Literal["hermetic_public", "external_private"]
    backend: str
    search_mode: KnowledgeSearchMode
    embedding_identity: str | None
    reranker_identity: str | None
    configuration: dict[str, Any]
    metrics: MemoryRetrievalBaselineMetrics
    language_slices: tuple[MemoryRetrievalLanguageSlice, ...]
    cases: tuple[MemoryRetrievalCaseResult, ...]

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, value: str) -> str:
        return require_durable_clean_nonblank(value, "backend")

    @field_validator("embedding_identity", "reranker_identity")
    @classmethod
    def validate_optional_component_identity(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("configuration", mode="before")
    @classmethod
    def copy_configuration(cls, value) -> dict[str, Any]:
        return copy_durable_json_object(value, "configuration")


def load_memory_retrieval_corpus(path: str | Path) -> MemoryRetrievalCorpus:
    corpus_path = Path(path)
    try:
        with corpus_path.open("rb") as stream:
            raw = stream.read(_MAX_CORPUS_BYTES + 1)
        if len(raw) > _MAX_CORPUS_BYTES:
            raise ValueError(f"Memory retrieval corpus exceeds {_MAX_CORPUS_BYTES} bytes.")
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to load memory retrieval corpus: {exc}") from exc
    return MemoryRetrievalCorpus.model_validate(payload)


async def run_memory_retrieval_baseline(
    corpus: MemoryRetrievalCorpus,
    store: KnowledgeStore,
    *,
    backend: str,
    search_mode: KnowledgeSearchMode = KnowledgeSearchMode.KEYWORD,
    embedding_identity: str | None = None,
    reranker_identity: str | None = None,
    configuration: dict[str, Any] | None = None,
) -> MemoryRetrievalBaselineResult:
    """Seed an empty store and measure the current knowledge-only memory baseline."""

    if type(corpus) is not MemoryRetrievalCorpus:
        raise TypeError("corpus must be a MemoryRetrievalCorpus.")
    corpus = MemoryRetrievalCorpus.model_validate(corpus.model_dump(mode="json"))
    if not isinstance(store, KnowledgeStore):
        raise TypeError("store must implement KnowledgeStore.")
    backend = require_durable_clean_nonblank(backend, "backend")
    if not isinstance(search_mode, KnowledgeSearchMode):
        raise TypeError("search_mode must be a KnowledgeSearchMode.")
    embedding_identity = (
        None
        if embedding_identity is None
        else require_durable_clean_nonblank(embedding_identity, "embedding_identity")
    )
    reranker_identity = (
        None
        if reranker_identity is None
        else require_durable_clean_nonblank(reranker_identity, "reranker_identity")
    )
    recorded_configuration = copy_durable_json_object(
        {} if configuration is None else configuration,
        "configuration",
    )
    required_configuration = {
        "cutoffs": sorted({case.cutoff for case in corpus.cases}),
        "source_set": ["knowledge"],
        "token_estimator": "ceil(utf8_json_bytes/4)",
    }
    for key, value in required_configuration.items():
        if key in recorded_configuration and recorded_configuration[key] != value:
            raise ValueError(f"configuration cannot override runner-owned field {key!r}.")
        recorded_configuration[key] = value
    privileged = KnowledgeAccessScope.privileged()
    existing = await store.list_entries(
        KnowledgeListQuery(
            statuses=list(KnowledgeStatus),
            include_expired=True,
            limit=1,
        ),
        access_scope=privileged,
    )
    if existing.entries:
        raise ValueError("Memory retrieval baseline requires an empty knowledge store.")
    entries_by_id = {entry.id: entry for entry in corpus.entries}
    for corpus_entry in corpus.entries:
        entry = corpus_entry.to_entry()
        await store.create_entry(
            entry,
            chunks=[
                KnowledgeChunk(
                    id=f"{entry.id}:r{entry.revision}:0",
                    entry_id=entry.id,
                    entry_revision=entry.revision,
                    chunk_index=0,
                    text=entry.text,
                )
            ],
            access_scope=privileged,
        )

    case_results: list[MemoryRetrievalCaseResult] = []
    selected_total = 0
    false_injections = 0
    stale_results = 0
    authorization_leaks = 0
    authorization_checks = 0
    citation_correct = 0
    citation_evaluated = 0
    for case in corpus.cases:
        scope = case.access.to_scope()
        started_ns = perf_counter_ns()
        result = await store.search(
            KnowledgeQuery(
                text=case.query,
                namespace=case.namespace,
                mode=search_mode,
                limit=case.cutoff,
            ),
            access_scope=scope,
        )
        latency_ms = (perf_counter_ns() - started_ns) / 1_000_000
        selected_ids = tuple(hit.entry.id for hit in result.hits[: case.cutoff])
        selected_total += len(selected_ids)
        relevant = set(case.relevant_entry_ids)
        selected_relevant = relevant & set(selected_ids)
        recall = len(selected_relevant) / len(relevant)
        first_rank = next(
            (index for index, entry_id in enumerate(selected_ids, start=1) if entry_id in relevant),
            None,
        )
        reciprocal_rank = 0.0 if first_rank is None else 1.0 / first_rank
        ndcg = _binary_ndcg(selected_ids, relevant, cutoff=case.cutoff)
        case_false_injections = sum(entry_id not in relevant for entry_id in selected_ids)
        case_stale = sum(
            entries_by_id[entry_id].status is not KnowledgeStatus.ACTIVE
            for entry_id in selected_ids
        )
        case_auth_leaks = len(set(selected_ids) & set(case.forbidden_entry_ids))
        authorization_checks += len(case.forbidden_entry_ids)
        case_citation_correct = 0
        case_citation_evaluated = 0
        for hit in result.hits[: case.cutoff]:
            expected_source = case.expected_source_ids.get(hit.entry.id)
            if expected_source is None:
                continue
            case_citation_evaluated += 1
            if hit.entry.source_id == expected_source:
                case_citation_correct += 1
        projection = [
            {
                "entry_id": hit.entry.id,
                "text_preview": hit.text_preview,
                "source_id": hit.entry.source_id,
                "reason": hit.reason,
            }
            for hit in result.hits[: case.cutoff]
        ]
        model_facing_bytes = len(
            canonical_durable_json_bytes(projection, "memory baseline model-facing projection")
        )
        candidate_count = result.total_hits_known
        if candidate_count is None:
            candidate_count = len(result.hits)
        false_injections += case_false_injections
        stale_results += case_stale
        authorization_leaks += case_auth_leaks
        citation_correct += case_citation_correct
        citation_evaluated += case_citation_evaluated
        case_results.append(
            MemoryRetrievalCaseResult(
                case_id=case.id,
                language=case.language,
                selected_entry_ids=selected_ids,
                candidate_count=candidate_count,
                truncated=result.truncated,
                recall_at_k=recall,
                reciprocal_rank=reciprocal_rank,
                ndcg_at_k=ndcg,
                false_injection_count=case_false_injections,
                stale_result_count=case_stale,
                authorization_leak_count=case_auth_leaks,
                citation_correct_count=case_citation_correct,
                citation_evaluated_count=case_citation_evaluated,
                model_facing_bytes=model_facing_bytes,
                latency_ms=latency_ms,
            )
        )

    for probe in corpus.id_probes:
        scope = probe.access.to_scope()
        for entry_id in probe.forbidden_entry_ids:
            authorization_checks += 2
            if await store.get_entry(entry_id, access_scope=scope) is not None:
                authorization_leaks += 1
            if await store.read_chunks(entry_id, access_scope=scope):
                authorization_leaks += 1

    latencies = [case.latency_ms for case in case_results]
    total_candidates = sum(case.candidate_count for case in case_results)
    model_facing_bytes = sum(case.model_facing_bytes for case in case_results)
    metrics = MemoryRetrievalBaselineMetrics(
        case_count=len(case_results),
        recall_at_k=_mean(case.recall_at_k for case in case_results),
        mean_reciprocal_rank=_mean(case.reciprocal_rank for case in case_results),
        mean_ndcg_at_k=_mean(case.ndcg_at_k for case in case_results),
        false_injection_rate=(false_injections / selected_total if selected_total else 0.0),
        stale_result_rate=(stale_results / selected_total if selected_total else 0.0),
        authorization_leak_rate=(
            authorization_leaks / authorization_checks if authorization_checks else 0.0
        ),
        citation_correctness=(citation_correct / citation_evaluated if citation_evaluated else 0.0),
        latency_p50_ms=_nearest_rank_percentile(latencies, 0.50),
        latency_p95_ms=_nearest_rank_percentile(latencies, 0.95),
        total_candidate_count=total_candidates,
        mean_candidate_count=total_candidates / len(case_results),
        truncated_case_count=sum(case.truncated for case in case_results),
        model_facing_bytes=model_facing_bytes,
        estimated_model_facing_tokens=ceil(model_facing_bytes / 4),
    )
    return MemoryRetrievalBaselineResult(
        schema_version=MEMORY_RETRIEVAL_BASELINE_SCHEMA_VERSION,
        corpus_revision=corpus.corpus_revision,
        corpus_origin=corpus.origin,
        backend=backend,
        search_mode=search_mode,
        embedding_identity=embedding_identity,
        reranker_identity=reranker_identity,
        configuration=recorded_configuration,
        metrics=metrics,
        language_slices=_language_slices(case_results),
        cases=tuple(case_results),
    )


def _binary_ndcg(selected_ids: tuple[str, ...], relevant: set[str], *, cutoff: int) -> float:
    dcg = sum(
        1.0 / log2(rank + 1)
        for rank, entry_id in enumerate(selected_ids[:cutoff], start=1)
        if entry_id in relevant
    )
    ideal_count = min(len(relevant), cutoff)
    ideal = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / ideal if ideal else 0.0


def _mean(values) -> float:
    copied = list(values)
    return sum(copied) / len(copied) if copied else 0.0


def _nearest_rank_percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(ceil(percentile * len(ordered)) - 1, 0)]


def _language_slices(
    cases: list[MemoryRetrievalCaseResult],
) -> tuple[MemoryRetrievalLanguageSlice, ...]:
    return tuple(
        MemoryRetrievalLanguageSlice(
            language=language,
            case_count=len(selected),
            recall_at_k=_mean(case.recall_at_k for case in selected),
            mean_reciprocal_rank=_mean(case.reciprocal_rank for case in selected),
            mean_ndcg_at_k=_mean(case.ndcg_at_k for case in selected),
        )
        for language in sorted({case.language for case in cases})
        if (selected := [case for case in cases if case.language == language])
    )


__all__ = [
    "MEMORY_RETRIEVAL_BASELINE_SCHEMA_VERSION",
    "MEMORY_RETRIEVAL_CORPUS_SCHEMA_VERSION",
    "MemoryRetrievalAccessSpec",
    "MemoryRetrievalBaselineMetrics",
    "MemoryRetrievalBaselineResult",
    "MemoryRetrievalCase",
    "MemoryRetrievalCaseResult",
    "MemoryRetrievalCorpus",
    "MemoryRetrievalCorpusEntry",
    "MemoryRetrievalIdProbe",
    "MemoryRetrievalLanguageSlice",
    "load_memory_retrieval_corpus",
    "run_memory_retrieval_baseline",
]
