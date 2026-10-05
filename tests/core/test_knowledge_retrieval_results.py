"""Bounded retrieval preserves previews, result metadata and detached records."""

import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import get_type_hints

import pytest

import cayu
from cayu._validation import canonical_durable_json_bytes
from cayu.knowledge import _retrieval_results as rules
from cayu.knowledge.indexing import KnowledgeIndexCoverage
from cayu.knowledge.records import KnowledgeChunk, KnowledgeEntry, KnowledgeEvidence
from cayu.knowledge.search import KnowledgeQuery

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _entry(id="entry"):
    return KnowledgeEntry(
        id=id, text="body", created_at=_NOW, updated_at=_NOW, metadata={"nested": ["original"]}
    )


def _chunk(index, text):
    return KnowledgeChunk(
        id=f"chunk-{index}",
        entry_id="entry",
        entry_revision=1,
        chunk_index=index,
        text=text,
        content_hash="original-hash",
        source_uri="https://example.com/source",
        metadata={"nested": ["original"]},
    )


@pytest.mark.parametrize(
    "budget,expected",
    [(-1, ""), (0, ""), (1, ""), (2, "é"), (4, "é"), (5, "é猫"), (6, "é猫x"), (20, "é猫x")],
)
def test_retrieval_utf8_budget_never_returns_a_partial_codepoint(budget, expected):
    assert rules._truncate_text_to_bytes("é猫x", budget) == expected
    assert rules._truncate_text_to_bytes("", budget) == ""


def test_retrieval_chunk_window_preserves_ties_order_and_identity():
    chunks = [_chunk(i, str(i)) for i in [4, 1, 3, 0]]
    selected = rules._center_chunk_window(chunks, chunk_index=2, max_chunks=2)
    assert [c.chunk_index for c in selected] == [1, 3]
    assert selected[0] is chunks[1] and selected[1] is chunks[2]
    assert [c.chunk_index for c in chunks] == [4, 1, 3, 0]
    assert rules._center_chunk_window(chunks, chunk_index=2, max_chunks=4) is chunks


@pytest.mark.parametrize(
    "start,end,count,budget,expected",
    [
        (0, None, 3, 7, ["a", "é猫", "z"]),
        (0, None, 3, 6, ["a", "é猫"]),
        (0, None, 3, 5, ["a", "é"]),
        (0, None, 3, 2, ["a"]),
        (1, 1, 3, 5, ["é猫"]),
        (0, None, 1, 7, ["a"]),
        (3, None, 3, 7, []),
    ],
)
def test_retrieval_chunk_limits_keep_ranges_copies_and_partial_hashes(
    start, end, count, budget, expected
):
    chunks = [_chunk(i, text) for i, text in enumerate(["a", "é猫", "z"])]
    snapshots = [c.model_dump() for c in chunks]
    selected = rules._bounded_chunks(
        chunks, start_index=start, end_index=end, max_chunks=count, max_bytes=budget
    )
    assert [c.text for c in selected] == expected
    for chunk in selected:
        original = chunks[chunk.chunk_index]
        assert chunk is not original
        assert chunk.id == original.id and chunk.source_uri == original.source_uri
        assert chunk.entry_id == original.entry_id and chunk.entry_revision == 1
        assert chunk.content_hash == ("original-hash" if chunk.text == original.text else None)
        chunk.metadata["nested"].append("changed")
    assert [c.model_dump() for c in chunks] == snapshots


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_chunks", 0),
        ("max_chunks", True),
        ("max_chunks", 1.5),
        ("max_bytes", -1),
        ("max_bytes", False),
        ("max_bytes", "5"),
    ],
)
def test_retrieval_chunk_limit_validation_precedes_empty_input(field, value):
    limits = {"max_chunks": 1, "max_bytes": 1, field: value}
    with pytest.raises(ValueError, match=field):
        rules._bounded_chunks([], start_index=0, end_index=None, **limits)


def test_retrieval_keeps_sqlite_chunk_limit_validation_at_its_existing_boundary():
    from cayu.storage import knowledge_sqlite

    # SQLite validates at its public read operation; its internal helper does not.
    assert (
        knowledge_sqlite._bounded_chunks(
            [], start_index=0, end_index=None, max_chunks=0, max_bytes=1
        )
        == []
    )
    with pytest.raises(ValueError, match="max_chunks"):
        rules._bounded_chunks([], start_index=0, end_index=None, max_chunks=0, max_bytes=1)


def test_retrieval_evidence_budget_counts_canonical_bytes_and_keeps_a_sorted_prefix():
    first = KnowledgeEvidence(
        id="a",
        entry_id="entry",
        source_type="document",
        source_id="source",
        source_revision="1",
        created_at=_NOW,
        locator={"page": 1},
        metadata={"nested": ["猫" * 10]},
    )
    second = KnowledgeEvidence(
        id="b",
        entry_id="entry",
        source_type="document",
        source_id="source",
        source_revision="1",
        created_at=_NOW,
    )
    first_size, second_size = [
        len(canonical_durable_json_bytes(e.model_dump(mode="json"), "evidence"))
        for e in [first, second]
    ]
    evidence = [second, first]
    for budget, ids in [
        (first_size - 1, []),
        (first_size, ["a"]),
        (first_size + second_size - 1, ["a"]),
        (first_size + second_size, ["a", "b"]),
    ]:
        assert [
            e.id
            for e in rules._bounded_knowledge_evidence(evidence, max_records=2, max_bytes=budget)
        ] == ids
    selected = rules._bounded_knowledge_evidence(
        evidence, max_records=1, max_bytes=first_size + second_size
    )
    assert selected == [first] and selected[0] is not first
    selected[0].metadata["nested"].append("changed")
    selected[0].locator["page"] = 2
    assert first.metadata["nested"] == ["猫" * 10] and first.locator == {"page": 1}
    assert evidence == [second, first]
    with pytest.raises(ValueError, match="max_records"):
        rules._bounded_knowledge_evidence([], max_records=True, max_bytes=1)
    with pytest.raises(ValueError, match="max_bytes"):
        rules._bounded_knowledge_evidence([], max_records=1, max_bytes=0)


@pytest.mark.parametrize("embedding", [False, True], ids=["keyword", "embedding"])
@pytest.mark.parametrize(
    "limit,budget,previews,complete,truncated",
    [
        (2, 1, [], [], True),
        (2, 2, ["é"], [False], True),
        (2, 5, ["é猫"], [True], True),
        (2, 6, ["é猫", "a"], [True, False], True),
        (2, 7, ["é猫", "ab"], [True, True], False),
        (1, 7, ["é猫"], [True], True),
    ],
)
def test_retrieval_search_results_preserve_order_metadata_and_preview_budgets(
    embedding, limit, budget, previews, complete, truncated
):
    entry, other = _entry(), _entry("other")
    chunk = _chunk(1, "é猫")
    query = KnowledgeQuery(text="query", limit=limit, max_bytes=budget)
    scored = [(4.0, entry, chunk, "first match", "é猫"), (9.0, other, None, "second match", "ab")]
    if embedding:
        result = rules._search_result_from_scored_embeddings(
            [(*row, 0.75, True) for row in scored], query, score_kind="hybrid"
        )
    else:
        result = rules._keyword_search_result_from_scored(scored, query, score_kind="keyword")
    assert [hit.text_preview for hit in result.hits] == previews
    assert [hit.text_preview_complete for hit in result.hits] == complete
    assert result.truncated is truncated and result.total_hits_known == 2
    assert result.limit == limit and result.max_bytes == budget
    assert result.query == query and result.query is not query
    assert result.index_coverage == []
    for rank, hit in enumerate(result.hits, start=1):
        row = scored[rank - 1]
        assert (hit.score, hit.entry.id, hit.reason, hit.rank) == (row[0], row[1].id, row[3], rank)
        assert hit.score_kind == ("hybrid" if embedding else "keyword")
        assert hit.score_normalized == (0.75 if embedding else None)
        assert hit.entry is not row[1]
        if hit.chunk is not None:
            assert hit.chunk == chunk and hit.chunk is not chunk
        hit.entry.metadata["nested"].append("changed")
    assert entry.metadata == other.metadata == {"nested": ["original"]}


@pytest.mark.parametrize("embedding", [False, True], ids=["keyword", "embedding"])
def test_retrieval_empty_candidates_and_empty_previews_stop_without_inventing_hits(embedding):
    query = KnowledgeQuery(text="query")
    builder = (
        rules._search_result_from_scored_embeddings
        if embedding
        else rules._keyword_search_result_from_scored
    )
    empty = builder([], query, score_kind="test")
    assert empty.hits == [] and empty.total_hits_known == 0 and not empty.truncated
    rows = [(1.0, _entry(), None, "empty", ""), (2.0, _entry("other"), None, "match", "text")]
    if embedding:
        rows = [(*row, None, True) for row in rows]
    result = builder(rows, query, score_kind="test")
    assert result.hits == [] and result.total_hits_known == 2 and result.truncated


def test_retrieval_embedding_result_retains_source_incompleteness_and_copied_coverage():
    coverage = KnowledgeIndexCoverage(
        projection_type="chunk_text",
        embedding_model="model",
        dimensions=2,
        preprocessing_version="v1",
        generator="generator",
        generator_version="v1",
        index_representation_version="v1",
        eligible_records=2,
        ready_records=1,
        pending_records=1,
        failed_records=0,
        high_water_sequence=3,
        complete=False,
    )
    result = rules._search_result_from_scored_embeddings(
        [(1.0, _entry(), None, "partial source", "text", None, False)],
        KnowledgeQuery(text="query"),
        score_kind="hybrid",
        index_coverage=[coverage],
    )
    assert result.truncated and not result.hits[0].text_preview_complete
    assert result.hits[0].score_normalized is None
    assert result.index_coverage == [coverage] and result.index_coverage[0] is not coverage


def test_retrieval_results_compose_without_storage_implementations():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from tests.core.test_knowledge_retrieval_results import _entry, KnowledgeQuery, rules
result = rules._keyword_search_result_from_scored(
    [(1.0, _entry(), None, "match", "text")], KnowledgeQuery(text="query"), score_kind="test")
assert result.hits[0].text_preview == "text"
assert not {"cayu.storage.memory", "cayu.storage.knowledge_sqlite", "cayu.storage.postgres"}.intersection(sys.modules)
""",
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_retrieval_results_have_one_owner_and_resolvable_annotations():
    from cayu.storage import knowledge_sqlite, memory, postgres

    for name in (
        "_center_chunk_window",
        "_bounded_chunks",
        "_bounded_knowledge_evidence",
        "_keyword_search_result_from_scored",
        "_search_result_from_scored_embeddings",
        "_truncate_text_to_bytes",
    ):
        canonical = getattr(rules, name)
        assert canonical.__module__ == rules.__name__
        assert not hasattr(memory, name)
        get_type_hints(canonical)
    for name in ("_center_chunk_window", "_truncate_text_to_bytes", "_bounded_knowledge_evidence"):
        assert not hasattr(knowledge_sqlite, name)
    assert knowledge_sqlite._retrieval_results is rules
    assert postgres._bounded_knowledge_evidence is rules._bounded_knowledge_evidence
    assert (
        postgres._search_result_from_scored_embeddings
        is rules._search_result_from_scored_embeddings
    )
