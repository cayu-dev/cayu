"""Activation decisions and receipt replay compose independently of storage."""

import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import get_type_hints

import pytest

import cayu
from cayu.knowledge import _activation_rules as rules
from cayu.knowledge import _revision_rules as revision
from cayu.knowledge.activation_contracts import (
    KnowledgeActivationAuthority,
    KnowledgeActivationDecision,
    KnowledgeActivationDisposition,
    KnowledgeActivationSource,
    KnowledgeGovernanceMode,
    prepare_knowledge_activation_request,
)
from cayu.knowledge.records import (
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeEvidence,
    KnowledgeStatus,
)
from cayu.knowledge.scopes import KnowledgeAccessScope

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_SCOPE = KnowledgeAccessScope.privileged()


def _material(kind="default"):
    before = KnowledgeEntry(
        id="entry",
        namespace="example",
        labels={"team": "sales"},
        text="Unicode café 🐈",
        status=KnowledgeStatus.PENDING,
        created_at=_NOW,
        updated_at=_NOW,
        metadata={"nested": {"values": ["original"]}},
    )
    chunk = (
        revision._default_chunk_for_entry(before)
        if kind == "default"
        else KnowledgeChunk(
            id="custom",
            entry_id="entry",
            text="part",
            chunk_index=4,
            content_hash="stored-hash",
            metadata={"nested": {"values": ["original"]}},
        )
    )
    chunks = [chunk]
    if kind == "multiple":
        chunks.append(chunk.model_copy(update={"id": "second", "chunk_index": 7}))
    evidence = [
        KnowledgeEvidence(
            id="evidence",
            entry_id="entry",
            chunk_id=chunk.id,
            source_type="document",
            source_id="source",
            source_revision="v1",
            created_at=_NOW,
            locator={"pages": [1]},
            metadata={"nested": {"values": ["original"]}},
        )
    ]
    request = prepare_knowledge_activation_request(
        before,
        chunks,
        evidence=evidence,
        access_scope=_SCOPE,
        operation_id="approve-entry",
        governance_mode=KnowledgeGovernanceMode.REVIEWED,
        source=KnowledgeActivationSource.REVIEW_APPROVAL,
        expected_revision=1,
    )
    authority = KnowledgeActivationAuthority(
        request=request,
        decision=KnowledgeActivationDecision(
            request_sha256=request.fingerprint,
            disposition=KnowledgeActivationDisposition.ACTIVATE,
            policy_identity="human-reviewer",
            policy_version="1",
            code="approved",
            annotations={"review": {"values": ["original"]}},
        ),
    )
    after = before.model_copy(
        update={
            "revision": 2,
            "status": KnowledgeStatus.ACTIVE,
            "updated_at": _NOW + timedelta(seconds=1),
        }
    )
    target = revision._copy_chunks_for_revision(chunks, after)
    copied = revision._copy_evidence_for_revision(
        evidence, entry=after, previous_chunks=chunks, chunks=target
    )
    publication, activation = rules._prepare_review_approval_receipts(
        before, after, target, copied, authority, committed_at=_NOW + timedelta(seconds=2)
    )
    return SimpleNamespace(
        before=before,
        after=after,
        chunks=target,
        evidence=copied,
        authority=authority,
        publication=publication,
        activation=activation,
    )


@pytest.mark.parametrize(
    "module", ["cayu.knowledge._activation_rules", "cayu.storage.knowledge_review"]
)
def test_activation_rules_and_review_adapter_compose_without_backends(module):
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib
import sys
importlib.import_module(sys.argv[1])
from tests.core.test_knowledge_activation_rules import _material, rules
m = _material("custom")
approval = rules._replay_review_approval_from_receipts(
    m.publication, m.activation, authority=m.authority
)
assert approval is not None and approval.entry == m.after and approval.receipt.replayed
assert not {
    "cayu.storage.memory", "cayu.storage.knowledge_sqlite", "cayu.storage.postgres",
}.intersection(sys.modules)
""",
            module,
        ],
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                (
                    str(Path(cayu.__file__).resolve().parent.parent),
                    str(Path(__file__).resolve().parents[2]),
                )
            ),
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_activation_rules_have_one_owner_and_resolvable_annotations():
    from cayu.storage import memory

    for name in (
        "_validate_review_approval_authority",
        "_validate_review_approval_scope",
        "_activation_receipt_matches",
        "_review_approval_publication_request_sha256",
        "_prepare_review_approval_receipts",
        "_review_approval_receipts_match",
        "_replay_review_approval_from_receipts",
    ):
        assert getattr(rules, name).__module__ == rules.__name__
        assert not hasattr(memory, name)
        get_type_hints(getattr(rules, name))


def test_review_approval_authority_requires_exact_mode_source_disposition_and_scope():
    m = _material()
    rules._validate_review_approval_authority(m.authority, access_scope=_SCOPE)
    invalid = [
        m.authority.model_copy(
            update={"request": m.authority.request.model_copy(update={field: value})}
        )
        for field, value in (
            ("mode", KnowledgeGovernanceMode.AUTONOMOUS),
            ("source", KnowledgeActivationSource.MODEL_TOOL),
            ("access_scope", KnowledgeAccessScope.for_namespace("example")),
        )
    ]
    invalid.append(
        m.authority.model_copy(
            update={
                "decision": m.authority.decision.model_copy(
                    update={"disposition": KnowledgeActivationDisposition.REJECT}
                )
            }
        )
    )
    for authority in invalid:
        with pytest.raises(ValueError, match="authority is invalid"):
            rules._validate_review_approval_authority(authority, access_scope=_SCOPE)


def test_review_approval_scope_preserves_namespace_and_label_restrictions():
    entry = _material().before
    rules._validate_review_approval_scope(entry, expected_namespace=None, expected_labels={})
    rules._validate_review_approval_scope(
        entry, expected_namespace="example", expected_labels={"team": "sales"}
    )
    for namespace, labels, message in (
        ("other", {}, "expected namespace"),
        (None, {"team": "other"}, "expected labels"),
        (None, {"missing": "value"}, "expected labels"),
    ):
        with pytest.raises(ValueError, match=message):
            rules._validate_review_approval_scope(
                entry, expected_namespace=namespace, expected_labels=labels
            )


@pytest.mark.parametrize(
    ("kind", "request_sha256"),
    [
        ("default", "a6dc4a8a70587f9b3a681a956a29f73b90b4d5833ca0283344b5e1fdc2c3a0ae"),
        ("custom", "89aaa1e6e86d1cb008a56918095a8d1ac9dac2dc2ab8fd765499b22a8171057c"),
        ("multiple", "4015578b505925ba1bc5e5eeee9f1ce632770c16748d0ef83a98922f3bee74cb"),
    ],
    ids=["default", "custom", "multiple"],
)
def test_review_approval_replay_reconstructs_exact_successor_and_detaches_material(
    kind, request_sha256
):
    m = _material(kind)
    assert m.publication.request_sha256 == request_sha256
    snapshots = [v.model_dump(mode="json") for v in (m.publication, m.activation, m.authority)]
    approval = rules._replay_review_approval_from_receipts(
        m.publication, m.activation, authority=m.authority
    )
    assert approval is not None
    assert approval.entry == m.after
    assert approval.receipt.replayed and not m.activation.replayed
    assert approval.receipt.committed_at == m.publication.committed_at
    assert approval.receipt.publication_request_sha256 == m.publication.request_sha256
    assert [
        v.model_dump(mode="json") for v in (m.publication, m.activation, m.authority)
    ] == snapshots
    approval.entry.metadata["nested"]["values"].append("changed")
    approval.receipt.authority.decision.annotations["review"]["values"].append("changed")
    approval.receipt.authority.request.evidence[0].locator["pages"].append(2)
    assert [
        v.model_dump(mode="json") for v in (m.publication, m.activation, m.authority)
    ] == snapshots


def test_review_approval_replay_rejects_tampered_receipts_and_authority():
    m = _material("custom")
    common = {
        "operation_id": "other",
        "entry_id": "other",
        "entry_revision": 3,
        "expected_revision": 2,
        "committed_at": _NOW + timedelta(seconds=3),
    }
    for receipt_name, updates in (
        ("publication", {**common, "request_sha256": "0" * 64, "entry_updated_at": _NOW}),
        ("activation", {**common, "publication_request_sha256": "0" * 64}),
    ):
        for field, value in updates.items():
            receipts = {"publication": m.publication, "activation": m.activation}
            receipts[receipt_name] = receipts[receipt_name].model_copy(update={field: value})
            assert (
                rules._replay_review_approval_from_receipts(**receipts, authority=m.authority)
                is None
            )
    for receipt_name in ("publication", "activation"):
        receipts = {"publication": m.publication, "activation": m.activation}
        receipts[receipt_name] = receipts[receipt_name].model_copy(update={"entry_revision": True})
        assert (
            rules._replay_review_approval_from_receipts(**receipts, authority=m.authority) is None
        )
    authority = m.authority.model_copy(
        update={"decision": m.authority.decision.model_copy(update={"policy_identity": "other"})}
    )
    assert (
        rules._replay_review_approval_from_receipts(
            m.publication, m.activation, authority=authority
        )
        is None
    )


def test_activation_receipt_matching_checks_identity_authority_and_commit_boundary():
    m = _material()
    kwargs = {
        "authority": m.authority,
        "publication_request_sha256": m.publication.request_sha256,
        "publication_committed_at": m.publication.committed_at,
    }
    assert rules._activation_receipt_matches(m.activation, **kwargs)
    for field, value in (
        ("operation_id", "other"),
        ("entry_id", "other"),
        ("entry_revision", 3),
        ("expected_revision", 2),
        ("publication_request_sha256", "0" * 64),
        ("committed_at", _NOW),
        ("entry_revision", True),
    ):
        assert not rules._activation_receipt_matches(
            m.activation.model_copy(update={field: value}), **kwargs
        )
    different = m.authority.model_copy(
        update={"decision": m.authority.decision.model_copy(update={"policy_identity": "other"})}
    )
    assert not rules._activation_receipt_matches(m.activation, **{**kwargs, "authority": different})


def test_review_approval_replay_normalizes_receipt_flags_without_mutating_inputs():
    m = _material()
    publication = m.publication.model_copy(update={"replayed": True})
    activation = m.activation.model_copy(update={"replayed": True})
    approval = rules._replay_review_approval_from_receipts(
        publication, activation, authority=m.authority
    )
    assert approval is not None and approval.entry == m.after and approval.receipt.replayed
    assert publication.replayed and activation.replayed
