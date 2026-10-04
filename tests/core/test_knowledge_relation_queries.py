"""Relation queries preserve revision meaning, cursor identity and page boundaries."""

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
from cayu._validation import canonical_durable_json_bytes
from cayu.knowledge import _relation_queries as rules
from cayu.knowledge.records import KnowledgeRevisionRef, KnowledgeStatus
from cayu.knowledge.relations import (
    MAX_KNOWLEDGE_RELATION_BYTES,
    KnowledgeLineageCurrentness,
    KnowledgeLineageQuery,
    KnowledgeLineageRole,
    KnowledgeRelation,
    KnowledgeRelationDirection,
    KnowledgeRelationKind,
    KnowledgeRelationQuery,
)
from cayu.knowledge.scopes import KnowledgeAccessScope

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_REFERENCE = KnowledgeRevisionRef(entry_id="replacement", revision=2)
_SOURCE = KnowledgeRevisionRef(entry_id="source", revision=1)
_SCOPE = KnowledgeAccessScope(allowed_namespaces=["example"], required_labels={"team": "sales"})


def _relation(index=0, *, kind=KnowledgeRelationKind.SUPERSEDES, metadata=None):
    return KnowledgeRelation(
        id=f"relation-{index:02d}",
        subject=_REFERENCE,
        object=_SOURCE
        if index == 0
        else KnowledgeRevisionRef(entry_id=f"source-{index}", revision=1),
        kind=kind,
        created_at=_NOW,
        policy_id="reviewed",
        metadata={} if metadata is None else metadata,
    )


def _link(relation, **overrides):
    return rules._knowledge_lineage_link(
        **{
            "relation_id": relation.id,
            "kind": relation.kind,
            "subject": relation.subject,
            "object_": relation.object,
            "created_at": relation.created_at,
            "reference": _REFERENCE,
            "subject_current": _REFERENCE,
            "subject_status": KnowledgeStatus.ACTIVE,
            "object_current": relation.object,
            "object_status": KnowledgeStatus.ACTIVE,
            **overrides,
        }
    )


def _query(family, **updates):
    cls = KnowledgeRelationQuery if family == "relation" else KnowledgeLineageQuery
    return cls(reference=_REFERENCE, **updates)


def _fingerprint(family, query, scope=_SCOPE, **kwargs):
    return getattr(rules, f"_knowledge_{family}_query_fingerprint")(query, scope, **kwargs)


def _page(family, query, candidates):
    fingerprint = _fingerprint(family, query)
    if family == "relation":
        return rules._bounded_knowledge_relation_result(query, candidates, fingerprint=fingerprint)
    return rules._bounded_knowledge_lineage_result(
        query,
        reference_current=_REFERENCE,
        reference_status=KnowledgeStatus.ACTIVE,
        candidates=candidates,
        fingerprint=fingerprint,
    )


def test_relation_queries_compose_without_storage_implementations():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from tests.core.test_knowledge_relation_queries import _link, _page, _query, _relation, rules
for family in ("relation", "lineage"):
    candidates = [_relation(i) for i in range(2)]
    if family == "lineage":
        candidates = [_link(item) for item in candidates]
    page = _page(family, _query(family, limit=1), candidates)
    assert page.truncated and page.next_cursor
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


def test_relation_queries_have_one_owner_and_resolvable_annotations():
    from cayu.storage import knowledge_sqlite, memory, postgres

    names = (
        "_KnowledgeRelationCursor",
        "_knowledge_relation_matches_lineage_query",
        "_knowledge_lineage_link",
        "_knowledge_relation_query_fingerprint",
        "_knowledge_lineage_query_fingerprint",
        "_encode_knowledge_relation_cursor",
        "_encode_knowledge_lineage_cursor",
        "_decode_knowledge_relation_cursor",
        "_decode_knowledge_lineage_cursor",
        "_bounded_knowledge_relation_result",
        "_bounded_knowledge_lineage_result",
    )
    for name in names:
        canonical = getattr(rules, name)
        assert canonical.__module__ == rules.__name__
        assert not hasattr(memory, name)
        get_type_hints(canonical)
        if name.startswith(("_bounded_", "_decode_")) or name in (
            "_knowledge_lineage_link",
            "_knowledge_relation_query_fingerprint",
            "_knowledge_lineage_query_fingerprint",
        ):
            assert getattr(knowledge_sqlite, name) is getattr(postgres, name) is canonical


@pytest.mark.parametrize("kind", list(KnowledgeRelationKind))
@pytest.mark.parametrize("direction", list(KnowledgeRelationDirection))
def test_lineage_matching_preserves_direction_symmetry_and_exact_revision(kind, direction):
    relation = _relation(kind=kind)
    for reference, endpoint in ((_REFERENCE, "subject"), (_SOURCE, "object")):
        query = KnowledgeLineageQuery(reference=reference, direction=direction)
        expected = (
            kind is KnowledgeRelationKind.CONTRADICTS
            or direction is KnowledgeRelationDirection.BOTH
            or (endpoint == "subject" and direction is KnowledgeRelationDirection.OUTGOING)
            or (endpoint == "object" and direction is KnowledgeRelationDirection.INCOMING)
        )
        assert rules._knowledge_relation_matches_lineage_query(relation, query) is expected
        stale = query.model_copy(update={"reference": reference.model_copy(update={"revision": 9})})
        assert not rules._knowledge_relation_matches_lineage_query(relation, stale)
        excluded = query.model_copy(
            update={"kinds": [k for k in KnowledgeRelationKind if k != kind]}
        )
        assert not rules._knowledge_relation_matches_lineage_query(relation, excluded)


@pytest.mark.parametrize(
    "kind,roles",
    [
        (
            KnowledgeRelationKind.SUPERSEDES,
            (KnowledgeLineageRole.SUPERSEDES, KnowledgeLineageRole.SUPERSEDED_BY),
        ),
        (
            KnowledgeRelationKind.DERIVED_FROM,
            (KnowledgeLineageRole.DERIVED_FROM, KnowledgeLineageRole.DERIVATION_SOURCE_FOR),
        ),
        (
            KnowledgeRelationKind.CONTRADICTS,
            (KnowledgeLineageRole.CONTRADICTS, KnowledgeLineageRole.CONTRADICTS),
        ),
    ],
)
def test_lineage_projection_preserves_roles_currentness_and_private_material(kind, roles):
    relation = _relation(kind=kind, metadata={"private": "not in projection"})
    for reference, counterpart, role in (
        (_REFERENCE, _SOURCE, roles[0]),
        (_SOURCE, _REFERENCE, roles[1]),
    ):
        link = _link(relation, reference=reference)
        assert link.role is role and link.counterpart == counterpart
        assert link.currentness is KnowledgeLineageCurrentness.CURRENT
        assert link.unresolved_contradiction is (kind is KnowledgeRelationKind.CONTRADICTS)
        assert set(link.model_dump()) == {
            "relation_id",
            "kind",
            "role",
            "counterpart",
            "counterpart_current",
            "counterpart_status",
            "currentness",
            "unresolved_contradiction",
            "created_at",
        }
        for field, current in (("subject_current", _REFERENCE), ("object_current", _SOURCE)):
            stale = _link(
                relation, reference=reference, **{field: current.model_copy(update={"revision": 9})}
            )
            assert stale.currentness is KnowledgeLineageCurrentness.STALE
            assert not stale.unresolved_contradiction
        for field in ("subject_status", "object_status"):
            assert not _link(
                relation, reference=reference, **{field: KnowledgeStatus.ARCHIVED}
            ).unresolved_contradiction
    with pytest.raises(ValueError, match="not a relation endpoint"):
        _link(relation, reference=KnowledgeRevisionRef(entry_id="other", revision=1))


@pytest.mark.parametrize("family", ["relation", "lineage"])
def test_query_fingerprints_bind_filters_and_access_but_allow_page_size_changes(family):
    query = _query(family)
    fingerprint = _fingerprint(family, query)
    assert (
        _fingerprint(
            family, _query(family, limit=1, max_bytes=MAX_KNOWLEDGE_RELATION_BYTES, cursor="opaque")
        )
        == fingerprint
    )
    for update in (
        {"reference": _SOURCE},
        {"direction": KnowledgeRelationDirection.INCOMING},
        {"kinds": [KnowledgeRelationKind.SUPERSEDES]},
    ):
        assert _fingerprint(family, query.model_copy(update=update)) != fingerprint
    for update in ({"allowed_namespaces": ["other"]}, {"required_labels": {"team": "other"}}):
        assert _fingerprint(family, query, _SCOPE.model_copy(update=update)) != fingerprint
    if family == "lineage":
        for update in (
            {"currentnesses": [KnowledgeLineageCurrentness.CURRENT]},
            {"counterpart_statuses": [KnowledgeStatus.ACTIVE]},
            {"unresolved_only": True},
        ):
            assert _fingerprint(family, query.model_copy(update=update)) != fingerprint
        assert (
            len({_fingerprint(family, query, through_change_sequence=s) for s in (None, 0, 1)}) == 3
        )
        for sequence in (-1, True, "1"):
            with pytest.raises((TypeError, ValueError)):
                _fingerprint(family, query, through_change_sequence=sequence)


@pytest.mark.parametrize("family", ["relation", "lineage"])
def test_query_cursors_preserve_exact_encoding_and_validate_before_use(family):
    query = _query(family)
    fingerprint = _fingerprint(family, query)
    relation = _relation()
    encode = getattr(rules, f"_encode_knowledge_{family}_cursor")
    decode = getattr(rules, f"_decode_knowledge_{family}_cursor")
    record = relation if family == "relation" else _link(relation)
    keyword = "relation" if family == "relation" else "link"
    encoded = encode(fingerprint=fingerprint, **{keyword: record})
    assert fingerprint == _GOLDEN[family]["fingerprint"]
    assert encoded == _GOLDEN[family]["cursor"]
    parsed = decode(encoded, fingerprint=fingerprint)
    assert parsed.created_at == _NOW and parsed.relation_id == relation.id and parsed.version == 1
    assert decode(None, fingerprint=fingerprint) is None
    with pytest.raises(ValueError, match="does not match"):
        decode(encoded, fingerprint="0" * 64)
    for token in (encoded + "=", "***", "☃", "e30", "A", "x" * 2049):
        with pytest.raises(ValueError):
            decode(token, fingerprint=fingerprint)
    payload = parsed.model_dump(mode="json")
    for update in (
        {"version": 2},
        {"fingerprint": fingerprint.upper()},
        {"created_at": "2026-01-01T00:00:00"},
        {"relation_id": ""},
        {"extra": "unknown"},
    ):
        malformed = (
            base64.urlsafe_b64encode(json.dumps({**payload, **update}).encode())
            .decode()
            .rstrip("=")
        )
        with pytest.raises(ValueError, match=f"Invalid knowledge {family} cursor"):
            decode(malformed, fingerprint=fingerprint)
    same_instant = record.model_copy(
        update={"created_at": _NOW.astimezone(timezone(timedelta(hours=6)))}
    )
    assert encode(fingerprint=fingerprint, **{keyword: same_instant}) == encoded


@pytest.mark.parametrize("family", ["relation", "lineage"])
def test_query_pages_preserve_budget_order_cursor_and_detached_results(family):
    relations = [
        _relation(i, metadata={"payload": "x" * 3000, "nested": ["original"]}) for i in range(48)
    ]
    candidates = relations if family == "relation" else [_link(item) for item in relations]
    attr = "relations" if family == "relation" else "links"
    query = _query(family, limit=1000, max_bytes=1_000_000)
    empty = _page(family, query, [])
    assert not getattr(empty, attr) and not empty.truncated and empty.next_cursor is None
    full = _page(family, query, candidates)
    assert getattr(full, attr) == candidates and not full.truncated and full.next_cursor is None
    for bounded in (
        _query(family, limit=2),
        _query(family, limit=1000, max_bytes=MAX_KNOWLEDGE_RELATION_BYTES),
    ):
        page = _page(family, bounded, candidates)
        selected = getattr(page, attr)
        assert 0 < len(selected) < len(candidates) and len(selected) <= bounded.limit
        assert selected == candidates[: len(selected)] and page.truncated and page.next_cursor
        used = sum(
            len(canonical_durable_json_bytes(item.model_dump(mode="json"), "result"))
            for item in selected
        )
        assert used <= bounded.max_bytes
        if len(selected) < bounded.limit:
            next_bytes = len(
                canonical_durable_json_bytes(
                    candidates[len(selected)].model_dump(mode="json"), "result"
                )
            )
            assert used + next_bytes > bounded.max_bytes
        cursor = getattr(rules, f"_decode_knowledge_{family}_cursor")(
            page.next_cursor, fingerprint=_fingerprint(family, bounded)
        )
        assert cursor.relation_id == relations[len(selected) - 1].id and cursor.created_at == _NOW
        assert selected[0] is not candidates[0] and page.query is not bounded
    with pytest.raises(ValueError, match="increasing page order"):
        _page(family, query, list(reversed(candidates)))
    with pytest.raises(ValueError, match="at least 8192"):
        _page(family, query.model_copy(update={"max_bytes": 1}), candidates)
    if family == "relation":
        relations[0].metadata["nested"].append("mutated")
        assert full.relations[0].metadata["nested"] == ["original"]
    else:
        assert full.links[0].counterpart is not candidates[0].counterpart


_GOLDEN = {
    "relation": {
        "fingerprint": "7460ab46d3da4bcc227170ecb03943bee39ff9f0302531336089a699ccc94fcd",
        "cursor": (
            "eyJjcmVhdGVkX2F0IjoiMjAyNi0wMS0wMVQwMDowMDowMFoiLCJmaW5nZXJwcmludCI6Ijc0NjBhYjQ2"
            "ZDNkYTRiY2MyMjcxNzBlY2IwMzk0M2JlZTM5ZmY5ZjAzMDI1MzEzMzYwODlhNjk5Y2NjOTRmY2QiLCJy"
            "ZWxhdGlvbl9pZCI6InJlbGF0aW9uLTAwIiwidmVyc2lvbiI6MX0"
        ),
    },
    "lineage": {
        "fingerprint": "7b7c2d5d79ffdd60e6b068aac39cb1865dd617f1ce6fa35ec153aff6c1f1807a",
        "cursor": (
            "eyJjcmVhdGVkX2F0IjoiMjAyNi0wMS0wMVQwMDowMDowMFoiLCJmaW5nZXJwcmludCI6IjdiN2MyZDVk"
            "NzlmZmRkNjBlNmIwNjhhYWMzOWNiMTg2NWRkNjE3ZjFjZTZmYTM1ZWMxNTNhZmY2YzFmMTgwN2EiLCJy"
            "ZWxhdGlvbl9pZCI6InJlbGF0aW9uLTAwIiwidmVyc2lvbiI6MX0"
        ),
    },
}
