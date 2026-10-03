"""Revision preparation composes without changing identity or resource boundaries."""

import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import get_type_hints

import pytest

import cayu
from cayu.knowledge import _revision_rules as rules
from cayu.knowledge.records import KnowledgeChunk, KnowledgeEntry, KnowledgeEvidence

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _entry():
    return KnowledgeEntry(
        id="entry",
        namespace="example",
        text="Unicode café 🐈",
        labels={"team": "sales"},
        source_uri="https://example.test/source",
        created_at=_NOW,
        updated_at=_NOW,
    )


def _chunk():
    return KnowledgeChunk(
        id="custom",
        entry_id="entry",
        entry_revision=1,
        text="part",
        chunk_index=4,
        content_hash="stored-hash",
        metadata={"nested": {"values": ["original"]}},
    )


def _evidence():
    return KnowledgeEvidence(
        id="evidence",
        entry_id="entry",
        entry_revision=1,
        chunk_id="custom",
        source_type="document",
        source_id="source",
        source_revision="v1",
        locator={"nested": {"pages": [1]}},
        metadata={"nested": {"values": ["original"]}},
        created_at=_NOW,
    )


def test_revision_rules_compose_without_storage_implementations():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from cayu.knowledge import _revision_rules as rules
from cayu.knowledge.records import KnowledgeEntry

before = KnowledgeEntry(id="entry", namespace="example", text="content")
after = before.model_copy(update={"revision": 2})
rules._validate_revision_successor(before, after)
chunk = rules._default_chunk_for_entry(before)
assert rules._has_only_default_chunk(before, [chunk])
assert rules._copy_chunks_for_revision([chunk], after)[0].id == "entry:r2:0"
assert rules._copy_evidence_for_revision([], entry=after, previous_chunks=[], chunks=[]) == []
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


def test_revision_rules_have_one_owner_and_resolvable_annotations():
    from cayu.storage import memory

    for name in (
        "_validate_revision_successor",
        "_copy_evidence_for_revision",
        "_copy_chunks_for_revision",
        "_default_chunk_for_entry",
        "_has_only_default_chunk",
    ):
        canonical = getattr(rules, name)
        assert canonical.__module__ == rules.__name__
        assert not hasattr(memory, name)
        get_type_hints(canonical)


def test_revision_successor_preserves_identity_and_timestamp_order():
    before = _entry()
    after = before.model_copy(update={"revision": 2})
    rules._validate_revision_successor(before, after)
    for update, message in (
        ({"id": "foreign"}, "logical entry id"),
        ({"namespace": "foreign"}, "logical namespace"),
        ({"created_at": _NOW + timedelta(seconds=1)}, "logical creation time"),
        ({"updated_at": _NOW - timedelta(seconds=1)}, "cannot move backwards"),
    ):
        with pytest.raises(ValueError, match=message):
            rules._validate_revision_successor(before, after.model_copy(update=update))


def test_revision_successor_relabel_requires_admitted_and_current_permission():
    from cayu._resource_access_errors import ResourceAccessDenied
    from cayu.knowledge import access
    from cayu.sessions.access import (
        SessionAccessRule,
        SessionAccessScope,
        SessionAccessSelector,
        _SessionAccessBounds,
    )

    before = _entry()
    after = before.model_copy(update={"revision": 2, "labels": {"team": "support"}})
    selector = SessionAccessRule(selectors=(SessionAccessSelector(key="team", values=("sales",)),))
    denied = SessionAccessScope(read=(selector,))
    allowed = SessionAccessScope(read=(selector,), relabel=(SessionAccessRule(allow_all=True),))
    for admitted, current in ((denied, denied), (allowed, denied), (denied, allowed)):
        token = access._bounds.set(_SessionAccessBounds(admitted, current))
        try:
            rules._validate_revision_successor(before, before.model_copy(update={"revision": 2}))
            with pytest.raises(ResourceAccessDenied):
                rules._validate_revision_successor(before, after)
            # Relabel denial still takes precedence over invalid logical identity.
            with pytest.raises(ResourceAccessDenied):
                rules._validate_revision_successor(before, after.model_copy(update={"id": "other"}))
        finally:
            access._bounds.reset(token)
    token = access._bounds.set(_SessionAccessBounds(allowed, allowed))
    try:
        rules._validate_revision_successor(before, after)
    finally:
        access._bounds.reset(token)


def test_default_revision_chunk_recognition_checks_every_persisted_field():
    entry = _entry()
    chunk = rules._default_chunk_for_entry(entry)
    assert chunk.id == "entry:r1:0"
    assert chunk.content_hash == "27a5f337b1c6b518c10f32a757938d39c359ca6a132fb0783ea84f0f5100a704"
    assert chunk.text == entry.text and chunk.source_uri == entry.source_uri
    assert rules._has_only_default_chunk(entry, [chunk])
    assert not rules._has_only_default_chunk(entry, [])
    assert not rules._has_only_default_chunk(entry, [chunk, chunk])
    for field, value in {
        "id": "other",
        "entry_id": "other",
        "entry_revision": 2,
        "text": "other",
        "chunk_index": 1,
        "content_hash": "other",
        "source_uri": None,
        "metadata": {"extra": 1},
    }.items():
        assert not rules._has_only_default_chunk(entry, [chunk.model_copy(update={field: value})])
    assert rules._copy_chunks_for_revision([], entry) == [chunk]


def test_revision_chunk_copies_preserve_content_order_and_detach_metadata():
    after = _entry().model_copy(update={"revision": 2, "text": "new entry text"})
    first = _chunk()
    second = first.model_copy(update={"id": "second", "chunk_index": 1})
    before_json = [item.model_dump(mode="json") for item in (first, second)]
    copied = rules._copy_chunks_for_revision([first, second], after)
    assert [item.id for item in copied] == ["entry:r2:4", "entry:r2:1"]
    assert all(item.entry_revision == 2 and item.content_hash == "stored-hash" for item in copied)
    assert [item.text for item in copied] == ["part", "part"]
    assert [item.model_dump(mode="json") for item in (first, second)] == before_json
    first.metadata["nested"]["values"].append("changed")
    assert copied[0].metadata == copied[1].metadata == {"nested": {"values": ["original"]}}
    copied[0].metadata["nested"]["values"].append("copy-only")
    assert copied[1].metadata == {"nested": {"values": ["original"]}}


def test_revision_evidence_copies_rebind_chunks_and_preserve_provenance():
    after = _entry().model_copy(update={"revision": 2})
    chunk = _chunk()
    evidence = _evidence()
    unlinked = evidence.model_copy(update={"id": "entry-only", "chunk_id": None})
    before_json = [item.model_dump(mode="json") for item in (evidence, unlinked)]
    chunks = rules._copy_chunks_for_revision([chunk], after)
    copied = rules._copy_evidence_for_revision(
        [unlinked, evidence], entry=after, previous_chunks=[chunk], chunks=chunks
    )
    assert [item.id for item in copied] == sorted(item.id for item in copied)
    linked = next(item for item in copied if item.chunk_id is not None)
    assert linked.id == "ke_68fc86bb04a1cbd207e43fc33653d8cd8c0fb3ab6a9e1230da5dbffe1a47a354"
    assert linked.chunk_id == "entry:r2:4" and linked.entry_revision == 2
    assert (
        linked.source_id == evidence.source_id
        and linked.source_revision == evidence.source_revision
    )
    assert linked.created_at == evidence.created_at and linked.locator == evidence.locator
    assert [item.model_dump(mode="json") for item in (evidence, unlinked)] == before_json
    assert (
        rules._copy_evidence_for_revision(
            [unlinked, evidence], entry=after, previous_chunks=[chunk], chunks=chunks
        )
        == copied
    )
    evidence.locator["nested"]["pages"].append(2)
    evidence.metadata["nested"]["values"].append("changed")
    assert linked.locator == {"nested": {"pages": [1]}}
    assert linked.metadata == {"nested": {"values": ["original"]}}


def test_revision_evidence_rejects_missing_chunk_references_and_duplicate_identity():
    after = _entry().model_copy(update={"revision": 2})
    chunk = _chunk()
    evidence = _evidence()
    target = rules._copy_chunks_for_revision([chunk], after)
    for previous, chunks in (
        ([], target),
        ([chunk], []),
        ([chunk], [target[0].model_copy(update={"chunk_index": 1})]),
    ):
        with pytest.raises(RuntimeError, match="unavailable source chunk"):
            rules._copy_evidence_for_revision(
                [evidence], entry=after, previous_chunks=previous, chunks=chunks
            )
    with pytest.raises(ValueError, match="ids must be unique"):
        rules._copy_evidence_for_revision(
            [evidence, evidence], entry=after, previous_chunks=[chunk], chunks=target
        )
