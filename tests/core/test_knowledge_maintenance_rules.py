"""Maintenance decisions preserve reviewed evidence and access boundaries."""

import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import get_type_hints

import pytest

import cayu
from cayu._validation import canonical_durable_json_bytes
from cayu.knowledge import _maintenance_rules as rules
from cayu.knowledge.maintenance_contracts import (
    KnowledgeMaintenanceProposal,
    KnowledgeMaintenanceStale,
)
from cayu.knowledge.records import (
    KnowledgeEntry,
    KnowledgeEvidence,
    KnowledgeEvidenceDisposition,
    KnowledgeEvidenceRole,
    KnowledgeRevisionRef,
    KnowledgeStatus,
    KnowledgeVisibility,
)
from cayu.knowledge.relations import KnowledgeRelation, KnowledgeRelationKind
from cayu.knowledge.scopes import KnowledgeAccessDenied, KnowledgeAccessScope

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _material():
    replacement = KnowledgeEntry(
        id="replacement",
        text="Reviewed café 🐈",
        namespace="example",
        labels={"team": "sales"},
        status=KnowledgeStatus.PENDING,
        created_at=_NOW,
        updated_at=_NOW + timedelta(seconds=2),
        metadata={"nested": {"values": ["original"]}},
    )
    sources = [
        replacement.model_copy(
            update={"id": name, "status": KnowledgeStatus.ACTIVE, "revision": index + 1}
        )
        for index, name in enumerate(("z-source", "a-source", "derived-source"))
    ]
    refs = [KnowledgeRevisionRef(entry_id=s.id, revision=s.revision) for s in sources]
    proposal = KnowledgeMaintenanceProposal(
        id="proposal",
        replacement=KnowledgeRevisionRef(entry_id=replacement.id, revision=1),
        sources=refs,
        relations=[
            KnowledgeRelation(
                id=f"relation-{index}",
                subject=KnowledgeRevisionRef(entry_id=replacement.id, revision=2),
                object=ref,
                kind=(
                    KnowledgeRelationKind.SUPERSEDES
                    if index < 2
                    else KnowledgeRelationKind.DERIVED_FROM
                ),
                policy_id="reviewed",
                created_at=_NOW,
            )
            for index, ref in enumerate(refs)
        ],
        access_scope=KnowledgeAccessScope.privileged(),
        policy_id="reviewed",
        created_at=_NOW,
        rationale="Consolidate reviewed knowledge.",
        evidence_summary="Exact source revisions.",
    )
    evidence = [
        KnowledgeEvidence(
            id=f"evidence-{source.id}",
            entry_id=replacement.id,
            source_type="knowledge_revision",
            source_id=source.id,
            source_revision=str(source.revision),
            source_hash=sha256(
                canonical_durable_json_bytes(source.model_dump(mode="json"), "source")
            ).hexdigest(),
            locator={"entry_id": source.id, "revision": source.revision},
            created_at=_NOW,
        )
        for source in sources
    ]
    return SimpleNamespace(
        replacement=replacement,
        sources=sources,
        proposal=proposal,
        evidence=evidence,
        current={entry.id: entry for entry in [replacement, *sources]},
    )


def test_maintenance_rules_compose_without_storage_implementations():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from tests.core.test_knowledge_maintenance_rules import _material, rules, _NOW
m = _material()
replacement, sources = rules._require_knowledge_maintenance_current_entries(
    m.proposal, m.current, access_scope=m.proposal.access_scope, operation="publish"
)
rules._require_knowledge_maintenance_publication_boundary(replacement, sources)
rules._require_knowledge_maintenance_source_evidence(m.evidence, sources)
active, archived = rules._knowledge_maintenance_successors(
    m.proposal, replacement, sources,
    access_scope=m.proposal.access_scope, committed_at=_NOW, operation="apply"
)
assert active.revision == 2 and [e.id for e in archived] == ["a-source", "z-source"]
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


def test_maintenance_rules_have_one_owner_and_resolvable_annotations():
    from cayu.storage import knowledge_postgres, knowledge_sqlite, memory

    for name in (
        "_require_knowledge_maintenance_current_entries",
        "_require_knowledge_maintenance_current_replacement",
        "_require_knowledge_maintenance_publication_boundary",
        "_require_knowledge_maintenance_source_evidence",
        "_knowledge_maintenance_successors",
    ):
        canonical = getattr(rules, name)
        assert canonical.__module__ == rules.__name__
        assert not hasattr(memory, name)
        assert getattr(knowledge_sqlite, name) is getattr(knowledge_postgres, name) is canonical
        get_type_hints(canonical)


@pytest.mark.parametrize("role", ["replacement", "source"])
@pytest.mark.parametrize("change", ["missing", "revision", "status"])
def test_maintenance_current_entries_reject_stale_reviewed_material(role, change):
    m = _material()
    target = m.replacement if role == "replacement" else m.sources[0]
    if change == "missing":
        del m.current[target.id]
    else:
        value = target.revision + 1 if change == "revision" else KnowledgeStatus.ARCHIVED
        m.current[target.id] = target.model_copy(update={change: value})
    with pytest.raises(KnowledgeMaintenanceStale) as exc:
        rules._require_knowledge_maintenance_current_entries(
            m.proposal, m.current, access_scope=m.proposal.access_scope, operation="apply"
        )
    assert exc.value.reason == f"{role}_{change}"


def test_maintenance_current_entries_check_authority_before_stale_details():
    m = _material()
    scope = KnowledgeAccessScope(
        allowed_namespaces=["example"],
        required_labels={"team": "sales"},
        allowed_statuses=[KnowledgeStatus.PENDING, KnowledgeStatus.ACTIVE],
    )
    # A mismatched proposal scope wins even when the current records are missing.
    with pytest.raises(KnowledgeAccessDenied) as exc:
        rules._require_knowledge_maintenance_current_entries(
            m.proposal, {}, access_scope=scope, operation="apply"
        )
    assert exc.value.operation == "apply"
    proposal = m.proposal.model_copy(update={"access_scope": scope})
    for target in [m.replacement, m.sources[0]]:
        current = dict(m.current)
        current[target.id] = target.model_copy(
            update={"labels": {"team": "private"}, "revision": target.revision + 1}
        )
        with pytest.raises(KnowledgeAccessDenied):
            rules._require_knowledge_maintenance_current_entries(
                proposal, current, access_scope=scope, operation="apply"
            )
    replacement, sources = rules._require_knowledge_maintenance_current_entries(
        proposal, m.current, access_scope=scope, operation="apply"
    )
    assert replacement is m.replacement
    assert [source.id for source in sources] == [ref.entry_id for ref in proposal.sources]
    assert all(source is m.current[source.id] for source in sources)


def test_maintenance_publication_preserves_every_source_boundary():
    m = _material()
    rules._require_knowledge_maintenance_publication_boundary(m.replacement, m.sources)
    for field, value in (
        ("namespace", "private"),
        ("labels", {"team": "private"}),
        ("visibility", KnowledgeVisibility.USER),
    ):
        changed = [*m.sources[:-1], m.sources[-1].model_copy(update={field: value})]
        with pytest.raises(ValueError, match="identical namespace, labels, and visibility"):
            rules._require_knowledge_maintenance_publication_boundary(m.replacement, changed)


def test_maintenance_evidence_binds_exact_content_and_source_set():
    m = _material()
    rules._require_knowledge_maintenance_source_evidence(list(reversed(m.evidence)), m.sources)
    for evidence, message in (
        (m.evidence[:-1], "exactly cover"),
        ([*m.evidence, m.evidence[0]], "cannot repeat"),
        ([*m.evidence, m.evidence[0].model_copy(update={"source_id": "other"})], "exactly cover"),
    ):
        with pytest.raises(ValueError, match=message):
            rules._require_knowledge_maintenance_source_evidence(evidence, m.sources)
    for field, value in (
        ("text", "Changed content"),
        ("metadata", {"nested": {"values": ["changed"]}}),
        ("updated_at", _NOW + timedelta(seconds=10)),
    ):
        changed = [m.sources[0].model_copy(update={field: value}), *m.sources[1:]]
        with pytest.raises(ValueError, match="does not bind"):
            rules._require_knowledge_maintenance_source_evidence(m.evidence, changed)


def test_maintenance_evidence_rejects_invalid_or_ambiguous_revision_material():
    m = _material()
    for update, message in (
        ({"source_type": "document"}, "live exact knowledge revision"),
        ({"source_id": None}, "live exact knowledge revision"),
        ({"source_revision": None}, "live exact knowledge revision"),
        ({"source_hash": None}, "live exact knowledge revision"),
        ({"chunk_id": "chunk"}, "live exact knowledge revision"),
        ({"role": KnowledgeEvidenceRole.SUPPORTING}, "live exact knowledge revision"),
        ({"disposition": KnowledgeEvidenceDisposition.DETACHED}, "live exact knowledge revision"),
        ({"source_revision": "invalid"}, "canonical integers"),
        ({"source_revision": "01"}, "canonical integers"),
        ({"source_revision": "+1"}, "canonical integers"),
        ({"source_hash": "0" * 64}, "does not bind"),
        ({"locator": {"entry_id": "other", "revision": 1}}, "does not bind"),
    ):
        evidence = [m.evidence[0].model_copy(update=update), *m.evidence[1:]]
        with pytest.raises(ValueError, match=message):
            rules._require_knowledge_maintenance_source_evidence(evidence, m.sources)


@pytest.mark.parametrize("commit_offset", [-10, 10])
def test_maintenance_successors_preserve_revisions_order_time_and_inputs(commit_offset):
    m = _material()
    before = [e.model_dump(mode="json") for e in [m.replacement, *m.sources]]
    scope = KnowledgeAccessScope(
        allowed_namespaces=["example"],
        required_labels={"team": "sales"},
        allowed_statuses=[KnowledgeStatus.PENDING, KnowledgeStatus.ACTIVE],
    )
    active, archived = rules._knowledge_maintenance_successors(
        m.proposal,
        m.replacement,
        m.sources,
        access_scope=scope,
        committed_at=_NOW + timedelta(seconds=commit_offset),
        operation="apply",
    )
    assert active.revision == 2 and active.status is KnowledgeStatus.ACTIVE
    # Retiring sources does not require permission to read archived records.
    assert [(e.id, e.revision, e.status) for e in archived] == [
        ("a-source", 3, KnowledgeStatus.ARCHIVED),
        ("z-source", 2, KnowledgeStatus.ARCHIVED),
    ]
    for successor in [active, *archived]:
        original = m.current[successor.id]
        assert successor is not original
        assert successor.updated_at == _NOW + timedelta(seconds=max(2, commit_offset))
        assert successor.model_dump(exclude={"revision", "status", "updated_at"}) == (
            original.model_dump(exclude={"revision", "status", "updated_at"})
        )
    assert before == [e.model_dump(mode="json") for e in [m.replacement, *m.sources]]
    for denied in (
        scope.model_copy(update={"allowed_statuses": [KnowledgeStatus.PENDING]}),
        scope.model_copy(update={"required_labels": {"team": "private"}}),
    ):
        with pytest.raises(KnowledgeAccessDenied):
            rules._knowledge_maintenance_successors(
                m.proposal,
                m.replacement,
                m.sources,
                access_scope=denied,
                committed_at=_NOW,
                operation="apply",
            )
