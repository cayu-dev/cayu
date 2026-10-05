"""Knowledge query interpretation preserves filtering, clocks and version limits."""

import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import get_type_hints

import pytest

import cayu
from cayu.knowledge import _query_rules as rules
from cayu.knowledge.changes import MAX_KNOWLEDGE_CHANGE_SEQUENCE
from cayu.knowledge.records import KnowledgeEntry
from cayu.knowledge.search import KnowledgeListQuery, KnowledgeQuery

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _entry(**updates):
    return KnowledgeEntry(
        **{
            "id": "entry",
            "text": "body",
            "namespace": "docs",
            "labels": {"project": "alpha", "extra": "retained"},
            "kind": "procedure",
            "source_type": "document",
            "source_id": "source",
            "aspects": ["a", "b"],
            "impact_targets": ["target-a"],
            "created_at": _NOW,
            "updated_at": _NOW,
            **updates,
        }
    )


@pytest.mark.parametrize("listing", [False, True], ids=["search", "list"])
@pytest.mark.parametrize(
    "changes,expected",
    [
        ({}, True),
        ({"namespace": "other"}, False),
        ({"labels": {"project": "other"}}, False),
        ({"labels": {"missing": "alpha"}}, False),
        ({"kinds": ["document"]}, False),
        ({"kinds": []}, False),
        ({"statuses": ["archived"]}, False),
        ({"visibilities": ["user"]}, False),
        ({"source_type": "other"}, False),
        ({"source_id": "other"}, False),
        ({"aspects": ["missing"]}, False),
        ({"aspects": ["missing", "a"]}, True),
        ({"impact_targets": ["missing"]}, False),
        ({"impact_targets": ["missing", "target-a"]}, True),
        ({"kinds": None, "visibilities": None, "aspects": [], "impact_targets": []}, True),
    ],
    ids=[
        "all-match",
        "namespace",
        "label-value",
        "label-key",
        "kind",
        "empty-kinds",
        "status",
        "visibility",
        "source-type",
        "source-id",
        "aspect-miss",
        "aspect-any",
        "target-miss",
        "target-any",
        "optional-filters",
    ],
)
def test_query_filters_preserve_matching_short_circuits_and_inputs(
    monkeypatch, listing, changes, expected
):
    calls = []

    class Clock:
        @staticmethod
        def now(tz):
            calls.append(tz)
            return _NOW - timedelta(microseconds=1)

    monkeypatch.setattr(rules, "datetime", Clock)
    entry = _entry(expires_at=_NOW)
    filters = {
        "namespace": "docs",
        "labels": {"project": "alpha"},
        "kinds": ["procedure"],
        "statuses": ["active"],
        "visibilities": ["global"],
        "source_type": "document",
        "source_id": "source",
        "aspects": ["b"],
        "impact_targets": ["target-a"],
        **changes,
    }
    query = KnowledgeListQuery(**filters) if listing else KnowledgeQuery(text="query", **filters)
    original = entry.model_dump(), query.model_dump()
    matcher = rules._entry_matches_list_query if listing else rules._entry_matches_query
    assert matcher(entry, query) is expected
    assert calls == ([UTC] if expected else [])
    assert (entry.model_dump(), query.model_dump()) == original


def test_query_list_namespace_can_be_unrestricted():
    assert rules._entry_matches_list_query(_entry(), KnowledgeListQuery())
    assert not rules._entry_matches_query(_entry(), KnowledgeQuery(text="query"))


@pytest.mark.parametrize(
    "groups,expected",
    [([["missing", "a"], ["b"]], True), ([["a"], ["missing"]], False), ([["missing"]], False)],
    ids=["any-within-all-between", "second-group-missing", "first-group-missing"],
)
def test_query_aspect_groups_require_each_group_without_mutating_inputs(groups, expected):
    query = KnowledgeQuery(namespace="docs", aspect_groups=groups)
    original = query.model_dump()
    assert rules._entry_matches_query(_entry(), query) is expected
    assert query.model_dump() == original


@pytest.mark.parametrize(
    "offset,include_expired,expected,reads",
    [
        (None, False, False, 0),
        (None, True, False, 0),
        (-1, False, True, 1),
        (0, False, True, 1),
        (1, False, False, 1),
        (-1, True, False, 0),
    ],
    ids=["no-expiry", "no-expiry-included", "past", "equal", "future", "include-expired"],
)
def test_query_expiry_preserves_boundary_and_clock_short_circuits(
    monkeypatch, offset, include_expired, expected, reads
):
    calls = []

    class Clock:
        @staticmethod
        def now(tz):
            calls.append(tz)
            return _NOW

    monkeypatch.setattr(rules, "datetime", Clock)
    expiry = None if offset is None else _NOW + timedelta(microseconds=offset)
    assert (
        rules._entry_is_expired(_entry(expires_at=expiry), include_expired=include_expired)
        is expected
    )
    assert calls == [UTC] * reads


@pytest.mark.parametrize("listing", [False, True], ids=["search", "list"])
def test_query_expiry_is_read_for_each_matching_entry(monkeypatch, listing):
    times = iter([_NOW - timedelta(microseconds=1), _NOW])

    class Clock:
        @staticmethod
        def now(tz):
            assert tz is UTC
            return next(times)

    monkeypatch.setattr(rules, "datetime", Clock)
    entry = _entry(expires_at=_NOW)
    query = (
        KnowledgeListQuery(namespace="docs")
        if listing
        else KnowledgeQuery(text="query", namespace="docs")
    )
    matcher = rules._entry_matches_list_query if listing else rules._entry_matches_query
    assert matcher(entry, query)
    assert not matcher(entry, query)
    assert next(times, None) is None
    query.include_expired = True
    assert matcher(entry, query)


@pytest.mark.parametrize(
    "fields,expected",
    [
        (
            {
                "text": "head",
                "any_terms": ["any-a", "any-b"],
                "all_terms": ["all-a"],
                "phrases": ["two  words"],
                "none_terms": ["private"],
            },
            "head any-a any-b all-a two  words",
        ),
        (
            {"any_terms": ["alpha", "beta"], "all_terms": ["alpha"], "phrases": ["two words"]},
            "alpha beta alpha two words",
        ),
        ({"text": "猫!?", "mode": "semantic", "none_terms": ["private"]}, "猫!?"),
        (
            {"all_terms": ["second", "first"], "phrases": ["tail phrase"]},
            "second first tail phrase",
        ),
    ],
    ids=["field-order", "cross-field-repetition", "raw-semantic", "without-free-text"],
)
def test_query_semantic_text_preserves_field_order_and_omits_negative_terms(fields, expected):
    query = KnowledgeQuery(**fields)
    original = query.model_dump()
    assert rules._semantic_query_text(query) == expected
    assert query.model_dump() == original


def test_query_semantic_text_rejects_metadata_only_queries():
    query = KnowledgeQuery(aspect_groups=[["a"]], none_terms=["private"])
    with pytest.raises(ValueError, match="semantic query text"):
        rules._semantic_query_text(query)


@pytest.mark.parametrize(
    "knowledge,index",
    [
        (None, None),
        (0, 0),
        (0, 7),
        (7, 0),
        (8, 4),
        (MAX_KNOWLEDGE_CHANGE_SEQUENCE, MAX_KNOWLEDGE_CHANGE_SEQUENCE),
    ],
    ids=[
        "absent",
        "zero",
        "index-only-progress",
        "knowledge-only-progress",
        "independent-progress",
        "maximum",
    ],
)
def test_query_frontier_accepts_paired_independent_version_limits(knowledge, index):
    assert rules._validate_knowledge_search_frontier(knowledge, index) is None


@pytest.mark.parametrize(
    "knowledge,index,message",
    [
        (None, 0, "must be supplied together"),
        (0, None, "must be supplied together"),
        (True, None, "must be supplied together"),
        (-1, 0, "knowledge_sequence"),
        (0, -1, "index_readiness_sequence"),
        (True, 0, "knowledge_sequence"),
        (0, False, "index_readiness_sequence"),
        ("1", 0, "knowledge_sequence"),
        (0, 1.0, "index_readiness_sequence"),
        (MAX_KNOWLEDGE_CHANGE_SEQUENCE + 1, 0, "knowledge_sequence"),
        (0, MAX_KNOWLEDGE_CHANGE_SEQUENCE + 1, "index_readiness_sequence"),
        (-1, False, "knowledge_sequence"),
    ],
    ids=[
        "missing-knowledge",
        "missing-index",
        "pair-before-type",
        "negative-knowledge",
        "negative-index",
        "bool-knowledge",
        "bool-index",
        "string-knowledge",
        "float-index",
        "knowledge-overflow",
        "index-overflow",
        "knowledge-first",
    ],
)
def test_query_frontier_preserves_validation_boundaries_and_order(knowledge, index, message):
    with pytest.raises(ValueError, match=message):
        rules._validate_knowledge_search_frontier(knowledge, index)


def test_query_rules_compose_without_storage_implementations():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from cayu.knowledge import _query_rules as rules
from cayu.knowledge.records import KnowledgeEntry
from cayu.knowledge.search import KnowledgeQuery
query = KnowledgeQuery(text="query")
assert rules._entry_matches_query(KnowledgeEntry(id="entry", text="body"), query)
assert rules._semantic_query_text(query) == "query"
rules._validate_knowledge_search_frontier(0, 0)
assert not {"cayu.storage.memory", "cayu.storage.knowledge_sqlite", "cayu.storage.postgres"}.intersection(sys.modules)
""",
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_query_rules_have_one_owner_and_resolvable_annotations():
    from cayu.storage import knowledge_sqlite, memory, postgres

    for name in (
        "_entry_matches_query",
        "_entry_matches_list_query",
        "_entry_matches_metadata",
        "_entry_is_expired",
        "_semantic_query_text",
        "_validate_knowledge_search_frontier",
    ):
        canonical = getattr(rules, name)
        assert canonical.__module__ == rules.__name__
        assert not hasattr(memory, name)
        get_type_hints(canonical)
    assert memory._query_rules is rules
    assert (
        knowledge_sqlite._validate_knowledge_search_frontier
        is rules._validate_knowledge_search_frontier
    )
    assert postgres._validate_knowledge_search_frontier is rules._validate_knowledge_search_frontier
    assert postgres._semantic_query_text is rules._semantic_query_text
