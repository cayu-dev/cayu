from __future__ import annotations

from cayu.storage import KnowledgeEntry, KnowledgeQuery, KnowledgeStore


async def assert_token_exact_phrase_search_conformance(store: KnowledgeStore) -> None:
    """Require structured phrases to match consecutive complete normalized tokens."""

    await store.create_entry(
        KnowledgeEntry(
            id="phrase_exact",
            text="A Cat dog pairing is token exact.",
        )
    )
    await store.create_entry(
        KnowledgeEntry(
            id="phrase_substrings",
            text="Only copycat dogma appears here.",
        )
    )
    await store.create_entry(
        KnowledgeEntry(
            id="phrase_reversed",
            text="The dog cat ordering is reversed.",
        )
    )
    await store.create_entry(
        KnowledgeEntry(
            id="phrase_across_fields",
            title="The copy cat",
            text="dog kennel notes.",
        )
    )
    phrase_result = await store.search(KnowledgeQuery(phrases=["cat dog"]))
    normalized_phrase_result = await store.search(KnowledgeQuery(phrases=["Cat, DOG!"]))
    assert [hit.entry.id for hit in phrase_result.hits] == ["phrase_exact"]
    assert [hit.entry.id for hit in normalized_phrase_result.hits] == ["phrase_exact"]
    assert phrase_result.hits[0].score > 0
    assert normalized_phrase_result.hits[0].score == phrase_result.hits[0].score

    any_result = await store.search(KnowledgeQuery(any_terms=["copycat"]))
    assert [hit.entry.id for hit in any_result.hits] == ["phrase_substrings"]

    all_result = await store.search(KnowledgeQuery(all_terms=["copycat dogma"]))
    assert [hit.entry.id for hit in all_result.hits] == ["phrase_substrings"]

    none_result = await store.search(KnowledgeQuery(any_terms=["copycat"], none_terms=["dogma"]))
    assert none_result.hits == []
