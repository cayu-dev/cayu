"""Backfill continuation binds its query and preserves deterministic page order."""

import base64
import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import get_type_hints

import pytest

import cayu
from cayu.knowledge import _embedding_backfill as rules
from cayu.knowledge.records import MAX_KNOWLEDGE_CHUNK_INDEX, KnowledgeChunk, KnowledgeStatus
from cayu.knowledge.scopes import KnowledgeAccessScope
from cayu.knowledge.search import KnowledgeListQuery

_NOW = datetime(2026, 1, 1, 0, 0, 0, 123456, tzinfo=UTC)
_QUERY = KnowledgeListQuery(namespace="example", labels={"team": "sales"})
_SCOPE = KnowledgeAccessScope(allowed_namespaces=["example"], required_labels={"team": "sales"})
_CHUNK = KnowledgeChunk(
    id="chunk-2", entry_id="entry", entry_revision=3, chunk_index=2, text="Text"
)


def _fingerprint(query=_QUERY, scope=_SCOPE, **updates):
    return rules._knowledge_embedding_backfill_fingerprint(
        query,
        scope,
        **{
            "refresh_existing": True,
            "embedding_model": "example-model",
            "embedding_dimensions": 3,
            **updates,
        },
    )


def _encode(**updates):
    return rules._encode_knowledge_embedding_backfill_cursor(
        **{
            "fingerprint": _fingerprint(),
            "importance": 0.375,
            "updated_at": _NOW,
            "chunk": _CHUNK,
            **updates,
        },
    )


def _decode(token, **updates):
    return rules._decode_knowledge_embedding_backfill_cursor(
        token, **{"fingerprint": _fingerprint(), **updates}
    )


def test_backfill_cursor_preserves_exact_encoding_and_timestamp_precision():
    token = _encode()
    assert _fingerprint() == _GOLDEN_FINGERPRINT
    assert token == _GOLDEN_CURSOR
    decoded = _decode(token)
    assert decoded.model_dump() == {
        "version": 1,
        "fingerprint": _GOLDEN_FINGERPRINT,
        "importance": 0.375,
        "updated_at": _NOW,
        "entry_id": "entry",
        "chunk_index": 2,
        "chunk_id": "chunk-2",
    }
    assert _encode(updated_at=_NOW.astimezone(timezone(timedelta(hours=6)))) == token
    assert _decode(None) is None
    assert _decode(token) is not decoded


def test_backfill_cursor_binds_query_scope_and_embedding_configuration():
    token = _encode()
    fingerprints = [
        _fingerprint(_QUERY.model_copy(update=update))
        for update in (
            {"namespace": "other"},
            {"labels": {"team": "other"}},
            {"statuses": [KnowledgeStatus.ARCHIVED]},
            {"include_expired": True},
            {"limit": 1},
        )
    ]
    fingerprints.extend(
        _fingerprint(scope=_SCOPE.model_copy(update=update))
        for update in ({"allowed_namespaces": ["other"]}, {"required_labels": {"team": "other"}})
    )
    fingerprints.extend(
        _fingerprint(**update)
        for update in (
            {"refresh_existing": False},
            {"embedding_model": "other-model"},
            {"embedding_dimensions": 4},
        )
    )
    assert len(set(fingerprints)) == len(fingerprints)
    for fingerprint in fingerprints:
        with pytest.raises(ValueError, match="does not match this query"):
            _decode(token, fingerprint=fingerprint)
    # Revalidation must also reject invalid mutable query/scope material.
    with pytest.raises(ValueError):
        _fingerprint(_QUERY.model_copy(update={"limit": 0}))
    with pytest.raises(ValueError):
        _fingerprint(scope=_SCOPE.model_copy(update={"allowed_namespaces": [""]}))


@pytest.mark.parametrize(
    "token",
    ["***", "☃", "e30", "A", "x" * 2049],
    ids=["invalid-base64", "non-ascii", "missing-fields", "incomplete-base64", "oversized"],
)
def test_backfill_cursor_rejects_malformed_tokens(token):
    with pytest.raises(ValueError):
        _decode(token)


def test_backfill_cursor_rejects_noncanonical_base64():
    token = _encode()
    with pytest.raises(ValueError, match="Invalid knowledge embedding backfill cursor"):
        _decode(token + "=")


@pytest.mark.parametrize(
    "update",
    [
        {"version": 2},
        {"fingerprint": "A" * 64},
        {"importance": True},
        {"importance": -0.1},
        {"importance": 1.1},
        {"updated_at": "2026-01-01T00:00:00"},
        {"entry_id": ""},
        {"chunk_index": -1},
        {"chunk_index": MAX_KNOWLEDGE_CHUNK_INDEX + 1},
        {"chunk_id": ""},
        {"extra": "unknown"},
    ],
    ids=[
        "version",
        "fingerprint",
        "boolean-importance",
        "negative-importance",
        "importance-range",
        "naive-time",
        "entry",
        "negative-index",
        "index-range",
        "chunk",
        "extra",
    ],
)
def test_backfill_cursor_rejects_invalid_payloads(update):
    payload = _decode(_encode()).model_dump(mode="json")
    token = (
        base64.urlsafe_b64encode(json.dumps({**payload, **update}).encode()).decode().rstrip("=")
    )
    with pytest.raises(ValueError, match="Invalid knowledge embedding backfill cursor"):
        _decode(token)


def test_backfill_order_preserves_all_tie_breakers_and_pre_epoch_microseconds():
    def key(**updates):
        return rules._knowledge_embedding_backfill_sort_key(
            **{
                "importance": 0.5,
                "updated_at": _NOW,
                "entry_id": "a",
                "chunk_index": 0,
                "chunk_id": "a",
                **updates,
            }
        )

    keys = [
        key(importance=0.75),
        key(updated_at=_NOW + timedelta(microseconds=1)),
        key(),
        key(entry_id="b"),
        key(entry_id="b", chunk_index=1),
        key(entry_id="b", chunk_index=1, chunk_id="b"),
        key(importance=0.25),
    ]
    assert sorted(reversed(keys)) == keys
    assert len(set(keys)) == len(keys)
    assert key(updated_at=_NOW.astimezone(timezone(timedelta(hours=-7)))) == key()
    before_epoch = datetime(1969, 12, 31, 23, 59, 59, 999998, tzinfo=UTC)
    assert key(updated_at=before_epoch + timedelta(microseconds=1)) < key(updated_at=before_epoch)
    parsed = _decode(_encode())
    assert rules._knowledge_embedding_backfill_sort_key(
        importance=parsed.importance,
        updated_at=parsed.updated_at,
        entry_id=parsed.entry_id,
        chunk_index=parsed.chunk_index,
        chunk_id=parsed.chunk_id,
    ) == key(importance=0.375, entry_id="entry", chunk_index=2, chunk_id="chunk-2")


def test_backfill_rules_compose_without_storage_implementations():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from tests.core.test_knowledge_embedding_backfill import _decode, _encode
assert _decode(_encode()).chunk_id == "chunk-2"
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


def test_backfill_rules_have_one_owner_and_resolvable_annotations():
    from cayu.storage import memory, postgres

    for name in (
        "_KnowledgeEmbeddingBackfillCursor",
        "_knowledge_embedding_backfill_fingerprint",
        "_encode_knowledge_embedding_backfill_cursor",
        "_decode_knowledge_embedding_backfill_cursor",
        "_knowledge_embedding_backfill_sort_key",
    ):
        canonical = getattr(rules, name)
        assert canonical.__module__ == rules.__name__
        assert not hasattr(memory, name)
        get_type_hints(canonical)
        if name not in (
            "_KnowledgeEmbeddingBackfillCursor",
            "_knowledge_embedding_backfill_sort_key",
        ):
            assert getattr(postgres, name) is canonical


_GOLDEN_FINGERPRINT = "41b2f8fe481bda6140b79aa8c1cd2c43134dbc7011eb13c7acf2b4a14d97f3d3"
_GOLDEN_CURSOR = "eyJjaHVua19pZCI6ImNodW5rLTIiLCJjaHVua19pbmRleCI6MiwiZW50cnlfaWQiOiJlbnRyeSIsImZpbmdlcnByaW50IjoiNDFiMmY4ZmU0ODFiZGE2MTQwYjc5YWE4YzFjZDJjNDMxMzRkYmM3MDExZWIxM2M3YWNmMmI0YTE0ZDk3ZjNkMyIsImltcG9ydGFuY2UiOjM3NWUtMywidXBkYXRlZF9hdCI6IjIwMjYtMDEtMDFUMDA6MDA6MDAuMTIzNDU2WiIsInZlcnNpb24iOjF9"
