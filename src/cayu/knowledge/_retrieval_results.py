"""Bounded search hits, chunk previews and evidence for knowledge retrieval."""

from __future__ import annotations

from cayu._validation import canonical_durable_json_bytes
from cayu.knowledge.indexing import KnowledgeIndexCoverage, copy_knowledge_index_coverage
from cayu.knowledge.records import (
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeEvidence,
    _validate_positive_int,
    copy_knowledge_chunk,
    copy_knowledge_evidence,
)
from cayu.knowledge.search import KnowledgeHit, KnowledgeQuery, KnowledgeSearchResult


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


def _keyword_search_result_from_scored(
    scored: list[tuple[float, KnowledgeEntry, KnowledgeChunk | None, str, str]],
    query: KnowledgeQuery,
    *,
    score_kind: str,
) -> KnowledgeSearchResult:
    hits: list[KnowledgeHit] = []
    remaining = query.max_bytes
    truncated = False
    for rank, (score, entry, chunk, reason, preview_text) in enumerate(
        scored[: query.limit], start=1
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
                score_kind=score_kind,
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


def _truncate_text_to_bytes(text: str, max_bytes: int) -> str:
    if max_bytes <= 0:
        return ""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")
