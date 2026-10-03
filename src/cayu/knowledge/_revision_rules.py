"""Shared preparation and invariants for knowledge revision successors."""

from __future__ import annotations

from hashlib import sha256

from cayu._validation import canonical_durable_json_bytes
from cayu.knowledge.records import (
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeEvidence,
    _copy_entry_evidence,
)


def _validate_revision_successor(
    current: KnowledgeEntry,
    successor: KnowledgeEntry,
) -> None:
    from cayu.knowledge.access import require_relabel

    require_relabel(current.labels, successor.labels)
    if successor.id != current.id:
        raise ValueError("Knowledge revision must preserve the logical entry id.")
    if successor.namespace != current.namespace:
        raise ValueError("Knowledge revision must preserve the logical namespace.")
    if successor.created_at != current.created_at:
        raise ValueError("Knowledge revision must preserve the logical creation time.")
    if successor.updated_at < current.updated_at:
        raise ValueError("Knowledge revision `updated_at` cannot move backwards.")


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
