from __future__ import annotations

import asyncio
from typing import cast

import pytest
from pydantic import ValidationError

from cayu.storage import (
    InMemoryKnowledgeStore,
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeIndexer,
    KnowledgeIndexRequest,
    KnowledgeIndexResult,
    KnowledgeQuery,
    SQLiteKnowledgeStore,
)


def test_knowledge_indexer_builds_heading_aware_chunks() -> None:
    text = """# Payments

Invoices above $10k require manager approval.

## Reminders

Do not send payment reminders when the PO number is missing.
"""

    result = KnowledgeIndexer().build(
        KnowledgeIndexRequest(
            text=text,
            entry_id="payments",
            title="Payment rules",
            kind="procedure",
            labels={"project": "invoice_agent"},
            source_type="file",
            source_uri="workspace://PAYMENTS.md",
            chunk_target_bytes=120,
            chunk_overlap_bytes=20,
        )
    )

    assert result.entry.id == "payments"
    assert result.entry.title == "Payment rules"
    assert result.entry.kind == "procedure"
    assert result.entry.source_hash == result.source_hash
    assert result.text_bytes == len(text.encode("utf-8"))
    assert result.chunk_count == len(result.chunks)
    assert len(result.chunks) >= 2
    assert result.chunks[0].id == "payments:0"
    assert result.chunks[0].content_hash is not None
    assert "Payments" in result.chunks[0].text
    assert result.chunks[-1].metadata["heading_paths"] == [["Payments", "Reminders"]]
    assert result.truncated is False


def test_knowledge_indexer_overlaps_split_chunks() -> None:
    text = "alpha " * 40 + "boundary fact " + "omega " * 40

    result = KnowledgeIndexer().build(
        KnowledgeIndexRequest(
            text=text,
            entry_id="overlap",
            chunk_target_bytes=80,
            chunk_overlap_bytes=24,
        )
    )

    assert len(result.chunks) > 1
    first_suffix = result.chunks[0].text[-12:]
    assert first_suffix.strip()
    assert first_suffix in result.chunks[1].text


def test_knowledge_indexer_writes_to_store_and_searches_chunks() -> None:
    async def run():
        store = InMemoryKnowledgeStore()
        indexer = KnowledgeIndexer(store)
        result = await indexer.index_text(
            KnowledgeIndexRequest(
                text="# Deploy\n\nRun migrations before deploy.\n\nCheck health after deploy.",
                entry_id="deploy",
                namespace="ops",
                kind="procedure",
                chunk_target_bytes=80,
                chunk_overlap_bytes=10,
            )
        )
        search = await store.search(KnowledgeQuery(text="health deploy", namespace="ops"))
        chunks = await store.read_chunks("deploy")
        return result, search, chunks

    result, search, chunks = asyncio.run(run())

    assert result.written is True
    assert result.unchanged is False
    assert [hit.entry.id for hit in search.hits] == ["deploy"]
    assert search.hits[0].chunk is not None
    assert len(chunks) == result.chunk_count


def test_knowledge_indexer_writes_to_sqlite_store(tmp_path) -> None:
    async def run():
        store = SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite")
        try:
            indexer = KnowledgeIndexer(store)
            result = await indexer.index_text(
                KnowledgeIndexRequest(
                    text="# Support\n\nEscalate refund disputes above $500.",
                    entry_id="support-refunds",
                    namespace="support",
                    labels={"team": "billing"},
                    kind="procedure",
                )
            )
            search = await store.search(
                KnowledgeQuery(
                    text="refund disputes",
                    namespace="support",
                    labels={"team": "billing"},
                )
            )
            return result, search
        finally:
            await store.close()

    result, search = asyncio.run(run())

    assert result.written is True
    assert [hit.entry.id for hit in search.hits] == ["support-refunds"]
    assert search.hits[0].chunk is not None


def test_knowledge_indexer_skips_unchanged_store_write() -> None:
    async def run():
        store = InMemoryKnowledgeStore()
        indexer = KnowledgeIndexer(store)
        request = KnowledgeIndexRequest(text="Stable policy text.", entry_id="stable")
        first = await indexer.index_text(request)
        second = await indexer.index_text(request)
        stored = await store.get_entry("stable")
        return first, second, stored

    first, second, stored = asyncio.run(run())

    assert first.written is True
    assert first.unchanged is False
    assert second.written is False
    assert second.unchanged is True
    assert stored is not None
    assert stored.source_hash == first.source_hash


def test_knowledge_indexer_rewrites_when_metadata_changes_with_same_text() -> None:
    async def run():
        store = InMemoryKnowledgeStore()
        indexer = KnowledgeIndexer(store)
        first = await indexer.index_text(
            KnowledgeIndexRequest(
                text="Same policy text.",
                entry_id="same",
                labels={"project": "old"},
                title="Old title",
                chunk_metadata={"version": "old"},
            )
        )
        second = await indexer.index_text(
            KnowledgeIndexRequest(
                text="Same policy text.",
                entry_id="same",
                labels={"project": "new"},
                title="New title",
                chunk_metadata={"version": "new"},
            )
        )
        entry = await store.get_entry("same")
        chunks = await store.read_chunks("same")
        return first, second, entry, chunks

    first, second, entry, chunks = asyncio.run(run())

    assert first.written is True
    assert second.written is True
    assert second.unchanged is False
    assert entry is not None
    assert entry.labels == {"project": "new"}
    assert entry.title == "New title"
    assert chunks[0].metadata["version"] == "new"


def test_knowledge_indexer_rewrites_when_store_has_stale_extra_chunks() -> None:
    async def run():
        store = InMemoryKnowledgeStore()
        indexer = KnowledgeIndexer(store)
        request = KnowledgeIndexRequest(text="Same policy text.", entry_id="stale-extra")
        built = indexer.build(request)
        await store.put_entry_with_chunks(
            built.entry,
            [
                *built.chunks,
                KnowledgeChunk(
                    id="stale-extra:1",
                    entry_id="stale-extra",
                    text="stale extra chunk",
                    chunk_index=1,
                    content_hash="sha256:stale",
                    metadata={"source_hash": built.source_hash},
                ),
            ],
        )
        result = await indexer.index_text(request)
        chunks = await store.read_chunks("stale-extra", max_chunks=10, max_bytes=1_000)
        return result, chunks

    result, chunks = asyncio.run(run())

    assert result.written is True
    assert result.unchanged is False
    assert [(chunk.chunk_index, chunk.text) for chunk in chunks] == [(0, "Same policy text.")]


def test_knowledge_indexer_truncates_at_max_chunks() -> None:
    text = "\n\n".join(f"Paragraph {index} has searchable content." for index in range(10))

    result = KnowledgeIndexer().build(
        KnowledgeIndexRequest(
            text=text,
            entry_id="truncated",
            chunk_target_bytes=45,
            chunk_overlap_bytes=5,
            max_chunks=2,
        )
    )

    assert result.truncated is True
    assert result.chunk_count == 2


def test_knowledge_indexer_keeps_overlap_inside_chunk_target() -> None:
    text = "\n\n".join(f"Paragraph {index} " + ("x" * 60) for index in range(4))

    result = KnowledgeIndexer().build(
        KnowledgeIndexRequest(
            text=text,
            entry_id="bounded-overlap",
            chunk_target_bytes=90,
            chunk_overlap_bytes=30,
        )
    )

    assert len(result.chunks) > 1
    for chunk in result.chunks:
        assert len(chunk.text.encode("utf-8")) <= 90


def test_knowledge_indexer_repeats_heading_context_for_split_block() -> None:
    text = "# Reminders\n\n" + ("Do not send payment reminders without a PO number. " * 8)

    result = KnowledgeIndexer().build(
        KnowledgeIndexRequest(
            text=text,
            entry_id="split-heading",
            chunk_target_bytes=95,
            chunk_overlap_bytes=20,
        )
    )

    assert len(result.chunks) > 1
    for chunk in result.chunks:
        assert chunk.text.startswith("Reminders\n\n")
        assert len(chunk.text.encode("utf-8")) <= 95
        assert chunk.metadata["heading_paths"] == [["Reminders"]]


def test_knowledge_indexer_keeps_heading_when_body_starts_with_heading_text() -> None:
    result = KnowledgeIndexer().build(
        KnowledgeIndexRequest(
            text="# Reminders\n\nReminders require operator approval.",
            entry_id="heading-prefix-collision",
            chunk_target_bytes=200,
            chunk_overlap_bytes=20,
        )
    )

    assert result.chunks[0].text == ("Reminders\n\nReminders require operator approval.")


def test_knowledge_indexer_reports_actual_inserted_overlap() -> None:
    text = ("a" * 40) + "\n\n" + ("b" * 80)

    result = KnowledgeIndexer().build(
        KnowledgeIndexRequest(
            text=text,
            entry_id="overlap-metadata",
            chunk_target_bytes=90,
            chunk_overlap_bytes=30,
        )
    )

    assert len(result.chunks) == 2
    assert result.chunks[1].metadata["overlap_from_previous_bytes"] == 8
    assert len(result.chunks[1].text.encode("utf-8")) == 90


def test_knowledge_indexer_validates_bounds_and_store_type() -> None:
    content_hash = "sha256:ed7002b439e9ac845f22357d822bac1444730fbdb6016d3ec9432297b9ec9f73"

    with pytest.raises(ValidationError, match="chunk_overlap_bytes"):
        KnowledgeIndexRequest(
            text="content",
            chunk_target_bytes=100,
            chunk_overlap_bytes=100,
        )
    with pytest.raises(ValidationError, match="at most half"):
        KnowledgeIndexRequest(
            text="content",
            chunk_target_bytes=100,
            chunk_overlap_bytes=51,
        )

    with pytest.raises(ValidationError, match="text"):
        KnowledgeIndexRequest(text=" ")
    with pytest.raises(ValidationError, match="at least 4"):
        KnowledgeIndexRequest(text="content", entry_text_max_bytes=1)
    with pytest.raises(ValidationError, match="at least 4"):
        KnowledgeIndexRequest(text="content", chunk_target_bytes=1)
    with pytest.raises(ValidationError, match="chunks"):
        KnowledgeIndexResult(
            entry=KnowledgeEntry(id="empty-result", text="content"),
            chunks=[],
            source_hash="sha256:abc",
            text_bytes=7,
            chunk_count=0,
        )
    with pytest.raises(ValidationError, match="written"):
        KnowledgeIndexResult(
            entry=KnowledgeEntry(
                id="conflicting-result",
                text="content",
                source_hash="sha256:abc",
            ),
            chunks=[
                KnowledgeChunk(
                    id="conflicting-result:0",
                    entry_id="conflicting-result",
                    text="content",
                    chunk_index=0,
                    content_hash=content_hash,
                    metadata={"source_hash": "sha256:abc"},
                )
            ],
            source_hash="sha256:abc",
            text_bytes=7,
            chunk_count=1,
            written=True,
            unchanged=True,
        )
    with pytest.raises(ValidationError, match="source_hash"):
        KnowledgeIndexResult(
            entry=KnowledgeEntry(
                id="source-mismatch",
                text="content",
                source_hash="sha256:entry",
            ),
            chunks=[
                KnowledgeChunk(
                    id="source-mismatch:0",
                    entry_id="source-mismatch",
                    text="content",
                    chunk_index=0,
                    content_hash=content_hash,
                    metadata={"source_hash": "sha256:result"},
                )
            ],
            source_hash="sha256:result",
            text_bytes=7,
            chunk_count=1,
        )
    with pytest.raises(ValidationError, match="chunks"):
        KnowledgeIndexResult(
            entry=KnowledgeEntry(
                id="chunk-owner",
                text="content",
                source_hash="sha256:abc",
            ),
            chunks=[
                KnowledgeChunk(
                    id="other:0",
                    entry_id="other",
                    text="content",
                    chunk_index=0,
                    content_hash=content_hash,
                    metadata={"source_hash": "sha256:abc"},
                )
            ],
            source_hash="sha256:abc",
            text_bytes=7,
            chunk_count=1,
        )
    with pytest.raises(ValidationError, match="text_bytes"):
        KnowledgeIndexResult(
            entry=KnowledgeEntry(
                id="text-bytes",
                text="content",
                source_hash="sha256:abc",
            ),
            chunks=[
                KnowledgeChunk(
                    id="text-bytes:0",
                    entry_id="text-bytes",
                    text="content",
                    chunk_index=0,
                    content_hash=content_hash,
                    metadata={"source_hash": "sha256:abc"},
                )
            ],
            source_hash="sha256:abc",
            text_bytes=6,
            chunk_count=1,
        )
    with pytest.raises(ValidationError, match="content_hash"):
        KnowledgeIndexResult(
            entry=KnowledgeEntry(
                id="bad-content-hash",
                text="content",
                source_hash="sha256:abc",
            ),
            chunks=[
                KnowledgeChunk(
                    id="bad-content-hash:0",
                    entry_id="bad-content-hash",
                    text="content",
                    chunk_index=0,
                    content_hash="sha256:wrong",
                    metadata={"source_hash": "sha256:abc"},
                )
            ],
            source_hash="sha256:abc",
            text_bytes=7,
            chunk_count=1,
        )
    with pytest.raises(ValidationError, match="metadata.source_hash"):
        KnowledgeIndexResult(
            entry=KnowledgeEntry(
                id="bad-chunk-source-hash",
                text="content",
                source_hash="sha256:abc",
            ),
            chunks=[
                KnowledgeChunk(
                    id="bad-chunk-source-hash:0",
                    entry_id="bad-chunk-source-hash",
                    text="content",
                    chunk_index=0,
                    content_hash=content_hash,
                    metadata={"source_hash": "sha256:wrong"},
                )
            ],
            source_hash="sha256:abc",
            text_bytes=7,
            chunk_count=1,
        )

    with pytest.raises(TypeError, match="KnowledgeStore"):
        KnowledgeIndexer(cast("InMemoryKnowledgeStore", object()))
