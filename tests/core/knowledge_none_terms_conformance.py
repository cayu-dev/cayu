from __future__ import annotations

from cayu.storage import (
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeQuery,
    KnowledgeSearchMode,
    KnowledgeStore,
)


async def assert_entry_wide_none_terms_conformance(
    store: KnowledgeStore,
    *,
    mode: KnowledgeSearchMode,
) -> None:
    cases = [
        (
            KnowledgeEntry(
                id="excluded_title",
                title="Deprecated archive",
                text="Current integration summary.",
            ),
            [
                KnowledgeChunk(
                    id="excluded_title:0",
                    entry_id="excluded_title",
                    chunk_index=0,
                    text="GitHub push credential instructions.",
                )
            ],
        ),
        (
            KnowledgeEntry(
                id="excluded_entry_text",
                text="Deprecated integration summary.",
            ),
            [
                KnowledgeChunk(
                    id="excluded_entry_text:0",
                    entry_id="excluded_entry_text",
                    chunk_index=0,
                    text="GitHub push credential instructions.",
                )
            ],
        ),
        (
            KnowledgeEntry(id="excluded_matching_chunk", text="Integration summary."),
            [
                KnowledgeChunk(
                    id="excluded_matching_chunk:0",
                    entry_id="excluded_matching_chunk",
                    chunk_index=0,
                    text="Deprecated GitHub push credential instructions.",
                )
            ],
        ),
        (
            KnowledgeEntry(id="excluded_sibling_chunk", text="Integration summary."),
            [
                KnowledgeChunk(
                    id="excluded_sibling_chunk:0",
                    entry_id="excluded_sibling_chunk",
                    chunk_index=0,
                    text="GitHub push credential instructions.",
                ),
                KnowledgeChunk(
                    id="excluded_sibling_chunk:1",
                    entry_id="excluded_sibling_chunk",
                    chunk_index=1,
                    text="Deprecated proxy guidance.",
                ),
                KnowledgeChunk(
                    id="excluded_sibling_chunk:2",
                    entry_id="excluded_sibling_chunk",
                    chunk_index=2,
                    text="Unrelated historical notes.",
                ),
            ],
        ),
        (
            KnowledgeEntry(id="safe_primary", text="Integration summary."),
            [
                KnowledgeChunk(
                    id="safe_primary:0",
                    entry_id="safe_primary",
                    chunk_index=0,
                    text="GitHub push credential instructions.",
                )
            ],
        ),
        (
            KnowledgeEntry(id="safe_later_chunk", text="Integration summary."),
            [
                KnowledgeChunk(
                    id="safe_later_chunk:0",
                    entry_id="safe_later_chunk",
                    chunk_index=0,
                    text="General release notes.",
                ),
                KnowledgeChunk(
                    id="safe_later_chunk:1",
                    entry_id="safe_later_chunk",
                    chunk_index=1,
                    text="GitHub GitHub push credential instructions.",
                ),
            ],
        ),
    ]
    for entry, chunks in cases:
        await store.create_entry(entry, chunks)
    if mode is not KnowledgeSearchMode.KEYWORD:
        process_embedding_changes = getattr(store, "process_embedding_changes", None)
        if process_embedding_changes is not None:
            await process_embedding_changes("none-terms-conformance", "worker")

    wide = await store.search(
        KnowledgeQuery(
            text="github",
            none_terms=["deprecated"],
            mode=mode,
            limit=20,
        )
    )

    assert {hit.entry.id for hit in wide.hits} == {"safe_primary", "safe_later_chunk"}
    assert wide.total_hits_known == 2
    assert wide.truncated is False
    later_hit = next(hit for hit in wide.hits if hit.entry.id == "safe_later_chunk")
    assert later_hit.chunk is not None
    assert later_hit.chunk.id == "safe_later_chunk:1"

    limited = await store.search(
        KnowledgeQuery(
            text="github",
            none_terms=["deprecated"],
            mode=mode,
            limit=1,
        )
    )

    assert len(limited.hits) == 1
    assert limited.hits[0].entry.id in {"safe_primary", "safe_later_chunk"}
    assert limited.total_hits_known == 2
    assert limited.truncated is True


async def assert_entry_wide_none_terms_precede_chunk_pagination(
    store: KnowledgeStore,
) -> None:
    await store.create_entry(
        KnowledgeEntry(id="excluded_many_chunks", text="Integration summary."),
        [
            *[
                KnowledgeChunk(
                    id=f"excluded_many_chunks:{index}",
                    entry_id="excluded_many_chunks",
                    chunk_index=index,
                    text=f"GitHub matching chunk {index}.",
                )
                for index in range(550)
            ],
            KnowledgeChunk(
                id="excluded_many_chunks:550",
                entry_id="excluded_many_chunks",
                chunk_index=550,
                text="Deprecated sibling guidance.",
            ),
        ],
    )
    await store.create_entry(KnowledgeEntry(id="safe_a", text="GitHub safe control A."))
    await store.create_entry(KnowledgeEntry(id="safe_b", text="GitHub safe control B."))

    result = await store.search(
        KnowledgeQuery(
            text="github",
            none_terms=["deprecated"],
            mode=KnowledgeSearchMode.KEYWORD,
            limit=2,
        )
    )

    assert {hit.entry.id for hit in result.hits} == {"safe_a", "safe_b"}
    assert result.total_hits_known == 2
    assert result.truncated is False
