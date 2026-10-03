"""Independent search/list contracts and compatibility with existing imports."""

import ast
import importlib
import inspect
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import get_type_hints

import pytest

import cayu

_PUBLIC_TYPES = (
    "KnowledgeSearchMode",
    "KnowledgeListGroup",
    "KnowledgeQuery",
    "KnowledgeListQuery",
    "KnowledgeHit",
    "KnowledgeSearchResult",
    "KnowledgeListItem",
    "KnowledgeFacet",
    "KnowledgeListResult",
)
_HELPERS = (
    "copy_knowledge_query",
    "copy_knowledge_list_query",
    "copy_knowledge_hit",
    "copy_knowledge_list_item",
    "copy_knowledge_facet",
    "_knowledge_query_terms",
    "_query_terms_have_positive_terms",
    "_normalize_search_term_groups",
    "_dedupe_search_term_groups",
    "_normalize_search_phrase",
    "_tokenize_search_text",
    "_expand_search_tokens",
    "_search_token_variants",
    "_plural_search_token",
    "_validate_nonnegative_float",
    "_validate_unit_float",
)
_CONSTANTS = (
    "MAX_KNOWLEDGE_QUERY_ASPECT_GROUPS",
    "MAX_KNOWLEDGE_QUERY_ASPECTS_PER_GROUP",
    "MAX_KNOWLEDGE_QUERY_GROUPED_ASPECTS",
    "MAX_KNOWLEDGE_QUERY_GROUPED_ASPECT_BYTES",
    "_SEARCH_TOKEN_RE",
)


@pytest.mark.parametrize("module_name", ("cayu", "cayu.storage", "cayu.knowledge.search"))
def test_search_contracts_compose_without_storage_implementations(module_name):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib
import sys

api = importlib.import_module(sys.argv[1])
from cayu.knowledge.records import KnowledgeChunk, KnowledgeEntry
from cayu.knowledge.search import _knowledge_query_terms, copy_knowledge_query

query = api.KnowledgeQuery(
    text="policies", all_terms=["delivery"], none_terms=["expired"],
    phrases=["same day"], aspect_groups=[["billing"]], labels={"team": "sales"},
)
assert _knowledge_query_terms(query) == {
    "any": ["policies", "policy"], "all": [["delivery", "deliveries"]],
    "none": ["expired", "expireds"], "phrases": [["same", "day"]],
}
copied = copy_knowledge_query(query)
assert copied == query and copied is not query
assert copied.labels is not query.labels
assert copied.aspect_groups[0] is not query.aspect_groups[0]
assert api.KnowledgeQuery(text="!!!", mode=api.KnowledgeSearchMode.SEMANTIC).text == "!!!"
assert api.KnowledgeQuery(aspect_groups=[["billing"]]).text is None
entry = KnowledgeEntry(id="entry", text="Same day delivery policies")
chunk = KnowledgeChunk(id="chunk", entry_id=entry.id, chunk_index=0, text=entry.text)
hit = api.KnowledgeHit(
    entry=entry, chunk=chunk, rank=1, score=2.0, score_normalized=0.5,
    text_preview=entry.text, text_preview_complete=True,
)
result = api.KnowledgeSearchResult(
    query=query, hits=[hit], limit=query.limit, max_bytes=query.max_bytes, total_hits_known=1,
)
query.labels["team"] = "changed"
query.aspect_groups[0].append("changed")
assert result.query.labels == {"team": "sales"}
assert result.query.aspect_groups == [["billing"]]
assert result.hits[0] is not hit and result.hits[0].entry is not hit.entry
assert result.hits[0].chunk is not hit.chunk
assert result.hits[0].text_preview_complete is True
assert "text_preview_complete" not in result.model_dump()["hits"][0]
listing = api.KnowledgeListQuery(group_by=api.KnowledgeListGroup.KIND)
item = api.KnowledgeListItem(entry=entry, chunk_count=1, text_preview=entry.text, text_preview_complete=True)
facet = api.KnowledgeFacet(field="kind", value=entry.kind, count=1)
page = api.KnowledgeListResult(
    query=listing, entries=[item], facets=[facet], limit=listing.limit,
    max_bytes=listing.max_bytes, total_entries_known=1,
)
assert page.entries[0] is not item and page.facets[0] is not facet
assert page.entries[0].entry is not item.entry
assert page.entries[0].text_preview_complete is True
assert "text_preview_complete" not in page.model_dump()["entries"][0]
assert not {"cayu.storage.memory", "cayu.storage.knowledge_sqlite", "cayu.storage.postgres"}.intersection(sys.modules)
""",
            module_name,
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_search_contracts_preserve_aliases_stubs_and_annotations():
    import cayu.storage as storage
    import cayu.storage.memory as legacy
    from cayu.knowledge import search

    for name in (*_PUBLIC_TYPES, "_SearchTerms", *_HELPERS, *_CONSTANTS):
        canonical = getattr(search, name)
        assert getattr(legacy, name) is canonical
        if name in _CONSTANTS:
            continue
        assert canonical.__module__ == search.__name__
        assert pickle.loads(f"ccayu.storage.memory\n{name}\n.".encode()) is canonical
        get_type_hints(canonical)
        if inspect.isclass(canonical):
            for method in vars(canonical).values():
                if isinstance(method, classmethod | staticmethod):
                    method = method.__func__
                elif isinstance(method, property):
                    method = method.fget
                if inspect.isfunction(method):
                    get_type_hints(method)
    for package in (cayu, storage):
        manifest = importlib.import_module(package.__name__ + "._exports").EXPORTS
        stub = ast.parse(Path(package.__file__).with_suffix(".pyi").read_text())
        imports = {
            alias.asname or alias.name: node.module
            for node in stub.body
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        for name in _PUBLIC_TYPES:
            assert manifest[name] == (search.__name__, name)
            assert imports[name] == search.__name__
            assert getattr(package, name) is getattr(search, name)


def test_search_values_round_trip_through_canonical_pickle_paths():
    from cayu.knowledge import search as api
    from cayu.knowledge.records import KnowledgeEntry

    entry = KnowledgeEntry(id="entry", text="Delivery policies")
    query = api.KnowledgeQuery(text="delivery", aspect_groups=[["billing"]])
    hit = api.KnowledgeHit(entry=entry, text_preview=entry.text, text_preview_complete=True)
    result = api.KnowledgeSearchResult(
        query=query, hits=[hit], limit=query.limit, max_bytes=query.max_bytes
    )
    listing = api.KnowledgeListQuery(group_by="kind")
    item = api.KnowledgeListItem(entry=entry, text_preview=entry.text, text_preview_complete=True)
    facet = api.KnowledgeFacet(field="kind", value=entry.kind, count=1)
    page = api.KnowledgeListResult(
        query=listing,
        entries=[item],
        facets=[facet],
        limit=listing.limit,
        max_bytes=listing.max_bytes,
    )
    for value in (
        query,
        hit,
        result,
        listing,
        item,
        facet,
        page,
        api.KnowledgeSearchMode.KEYWORD,
        api.KnowledgeListGroup.KIND,
    ):
        encoded = pickle.dumps(value)
        assert api.__name__.encode() in encoded
        restored = pickle.loads(encoded)
        assert type(restored) is type(value)
        assert restored == value
    assert pickle.loads(pickle.dumps(result)).hits[0].text_preview_complete is True
    assert pickle.loads(pickle.dumps(page)).entries[0].text_preview_complete is True
