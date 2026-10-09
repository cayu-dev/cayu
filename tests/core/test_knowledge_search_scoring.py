"""Shared search scoring preserves term, phrase and best-match semantics."""

import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import get_type_hints

import pytest

import cayu
from cayu.knowledge import _search_scoring as rules
from cayu.knowledge.records import KnowledgeChunk, KnowledgeEntry
from cayu.knowledge.search import KnowledgeQuery, _knowledge_query_terms

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _entry(text, *, title=None):
    return KnowledgeEntry(id="entry", text=text, title=title, created_at=_NOW, updated_at=_NOW)


def _chunk(text, *, id="chunk", index=0):
    return KnowledgeChunk(id=id, entry_id="entry", entry_revision=1, text=text, chunk_index=index)


@pytest.mark.parametrize(
    "text,query,expected",
    [
        ("unrelated", {"aspect_groups": [["topic"]]}, 0.0),
        ("Docs docs", {"text": "doc"}, 2.0),
        ("policies policy", {"all_terms": ["policy"]}, 1.0),
        ("STRASSE", {"text": "Straße"}, 1.0),
        ("cat catalog", {"text": "cat"}, 1.0),
        ("alpha beta gamma beta", {"any_terms": ["alpha"], "all_terms": ["beta gamma"]}, 4.0),
        ("beta gamma", {"any_terms": ["alpha"], "all_terms": ["beta gamma"]}, 0.0),
        ("alpha beta", {"any_terms": ["alpha"], "all_terms": ["beta gamma"]}, 0.0),
        ("alpha secret", {"text": "alpha", "none_terms": ["secrets"]}, 0.0),
        ("alpha beta alpha beta", {"phrases": ["alpha beta", "gamma delta"]}, 2.0),
        ("alpha gamma beta", {"phrases": ["alpha beta"]}, 0.0),
        ("alpha beta", {"text": "alpha", "phrases": ["alpha beta"]}, 3.0),
    ],
    ids=[
        "aspect-only",
        "plural-any",
        "plural-all",
        "unicode-casefold",
        "whole-token",
        "combined-groups",
        "missing-any",
        "missing-all",
        "excluded-variant",
        "phrase-alternatives",
        "nonadjacent-phrase",
        "term-and-phrase",
    ],
)
def test_search_candidate_preserves_structured_matching_and_scores(text, query, expected):
    terms = _knowledge_query_terms(KnowledgeQuery(**query))
    assert rules._score_candidate(text, terms) == expected


def test_search_phrases_stay_within_title_entry_or_chunk_fields():
    entry = _entry("fox", title="red")
    query = KnowledgeQuery(phrases=["red fox"])
    terms = _knowledge_query_terms(query)
    assert rules._score_candidate("red\nfox", terms) == 2.0
    assert rules._score_candidate("red\nfox", terms, phrase_fields=["red", "fox"]) == 0.0
    assert rules._score_entry(entry, [_chunk("tail")], query) == (
        0.0,
        None,
        "entry text match",
        "fox",
    )
    chunk = _chunk("red fox")
    assert rules._score_entry(entry, [chunk], query) == (2.0, chunk, "chunk text match", "red fox")
    assert rules._score_entry(_entry("red"), [_chunk("fox")], query) == (
        0.0,
        None,
        "entry text match",
        "red",
    )


def test_search_best_match_preserves_title_weight_context_ties_and_preview():
    query = KnowledgeQuery(text="alpha")
    entry = _entry("unrelated", title="alpha")
    duplicate_title = _chunk("unrelated")
    assert rules._score_entry(entry, [duplicate_title], query) == (
        1.2,
        None,
        "title match",
        "alpha",
    )
    first = _chunk("alpha", id="first")
    second = _chunk("alpha", id="second", index=1)
    assert rules._score_entry(entry, [first, second], query) == (
        2.0,
        first,
        "chunk text match",
        "alpha",
    )
    # Equal chunk scores retain the first candidate, including the selected object.
    assert rules._score_entry(entry, [second, first], query)[1] is second
    body = _entry("alpha")
    assert rules._score_entry(body, [_chunk("alpha")], query) == (
        1.0,
        None,
        "entry text match",
        "alpha",
    )
    assert rules._entry_chunk_searchable_fields(entry, duplicate_title) == ["alpha", "unrelated"]
    assert rules._score_entry(
        body, [first], KnowledgeQuery(none_terms=["secret"], aspect_groups=[["topic"]])
    ) == (
        0.0,
        None,
        "empty query",
        "alpha",
    )


@pytest.mark.parametrize("field", ["title", "entry", "chunk"])
def test_search_exclusions_cover_the_complete_entry_without_mutation(field):
    entry = _entry(
        "secret" if field == "entry" else "alpha", title="secret" if field == "title" else "title"
    )
    chunks = [_chunk("secret" if field == "chunk" else "chunk")]
    query = KnowledgeQuery(text="alpha", none_terms=["secrets"])
    snapshots = entry.model_dump(), chunks[0].model_dump(), query.model_dump()
    terms = _knowledge_query_terms(query)
    assert rules._entry_matches_none_terms(entry, chunks, terms)
    rules._score_entry(entry, chunks, query)
    assert (entry.model_dump(), chunks[0].model_dump(), query.model_dump()) == snapshots
    assert not rules._entry_matches_none_terms(
        entry, chunks, _knowledge_query_terms(KnowledgeQuery(text="alpha"))
    )
    assert not rules._entry_matches_none_terms(_entry("secretary"), [], terms)


def test_search_scoring_composes_without_storage_implementations():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from tests.core.test_knowledge_search_scoring import _entry, KnowledgeQuery, rules
assert rules._score_entry(_entry("alpha"), [], KnowledgeQuery(text="alpha"))[0] == 1.0
assert not {
    "cayu.storage.memory", "cayu.storage.knowledge_sqlite", "cayu.storage.postgres",
}.intersection(sys.modules)
""",
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_search_scoring_has_one_owner_and_resolvable_annotations():
    from cayu.storage import knowledge_embedding_postgres, memory

    for name in (
        "_score_entry",
        "_score_candidate",
        "_tokens_match_structured_terms",
        "_tokens_contain_phrase",
        "_entry_chunk_searchable_fields",
        "_entry_matches_none_terms",
    ):
        canonical = getattr(rules, name)
        assert canonical.__module__ == rules.__name__
        assert not hasattr(memory, name)
        get_type_hints(canonical)
    assert knowledge_embedding_postgres._score_entry is rules._score_entry
