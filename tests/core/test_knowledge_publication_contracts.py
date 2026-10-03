"""Independent publication preparation and compatibility with existing imports."""

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

_OWNERS = {
    "scopes": (
        "_KnowledgeAccessSnapshot",
        "_knowledge_access_snapshot",
        "_knowledge_access_snapshot_json",
        "_parse_knowledge_access_snapshot_json",
    ),
    "activation_contracts": (
        "_MAX_KNOWLEDGE_ACTIVATION_RETIREMENT_BYTES",
        "_MAX_KNOWLEDGE_ACTIVATION_RETIREMENT_TIME",
        "_KnowledgeActivationRetirement",
        "_knowledge_activation_retirement",
        "_knowledge_activation_retirement_json",
        "_parse_knowledge_activation_retirement_json",
        "_require_knowledge_activation_retirement_capacity",
    ),
    "publication_contracts": (
        "KnowledgePublicationConflict",
        "KnowledgePublicationReceipt",
        "copy_knowledge_publication_receipt",
        "prepare_knowledge_publication",
        "_validate_knowledge_publication_replay",
        "_validate_activation_publication_material",
        "_knowledge_publication_request_sha256",
        "_knowledge_publication_v1_request_sha256",
        "_validate_revision_append",
    ),
}
_EXPORTS = (
    "KnowledgePublicationConflict",
    "KnowledgePublicationReceipt",
    "prepare_knowledge_publication",
)


@pytest.mark.parametrize(
    "module_name", ("cayu", "cayu.storage", "cayu.knowledge.publication_contracts")
)
def test_publication_contracts_compose_without_storage_implementations(module_name):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib
import sys
from datetime import UTC, datetime

from cayu.knowledge.activation_contracts import (
    KnowledgeActivationAuthority, KnowledgeActivationDecision,
    prepare_knowledge_activation_request,
)
from cayu.knowledge.publication_contracts import (
    _knowledge_publication_v1_request_sha256, _validate_knowledge_publication_replay,
    copy_knowledge_publication_receipt,
)
from cayu.knowledge.records import KnowledgeChunk, KnowledgeEntry, KnowledgeEvidence
from cayu.knowledge.scopes import KnowledgeAccessScope

api = importlib.import_module(sys.argv[1])
now = datetime(2026, 1, 1, tzinfo=UTC)
entry = KnowledgeEntry(id="entry", text="Exact publication", created_at=now, updated_at=now,
                       metadata={"nested": ["entry"]})
chunk = KnowledgeChunk(id="chunk", entry_id=entry.id, chunk_index=0, text=entry.text,
                       metadata={"nested": ["chunk"]})
evidence = KnowledgeEvidence(id="evidence", entry_id=entry.id, chunk_id=chunk.id,
                            source_type="test", source_id="source", source_revision="1",
                            metadata={"nested": ["evidence"]}, created_at=now)
scope = KnowledgeAccessScope.privileged()
request = prepare_knowledge_activation_request(
    entry, [chunk], evidence=[evidence], access_scope=scope, operation_id="operation",
    governance_mode="policy_automatic", source="curator",
)
authority = KnowledgeActivationAuthority(request=request, decision=KnowledgeActivationDecision(
    request_sha256=request.fingerprint, disposition="activate", policy_identity="app-policy",
    policy_version="1", code="approved",
))
digests = []
for items, proof in (([], None), ([evidence], None), ([evidence], authority)):
    operation, copied, chunks, copied_evidence, digest = api.prepare_knowledge_publication(
        entry, [chunk], evidence=items, activation_authority=proof, operation_id="operation",
    )
    assert operation == "operation"
    assert copied == entry and copied is not entry
    assert copied.metadata["nested"] is not entry.metadata["nested"]
    assert chunks[0] == chunk and chunks[0].metadata["nested"] is not chunk.metadata["nested"]
    if items:
        assert copied_evidence[0] is not evidence
        assert copied_evidence[0].metadata["nested"] is not evidence.metadata["nested"]
    receipt = api.KnowledgePublicationReceipt(
        operation_id=operation, entry_id=copied.id, entry_revision=1, expected_revision=None,
        request_sha256=digest, entry_created_at=now, entry_updated_at=now, committed_at=now,
    )
    replay = copy_knowledge_publication_receipt(receipt, replayed=True)
    assert replay is not receipt and replay.replayed and not receipt.replayed
    kwargs = dict(entry=copied, chunks=chunks, evidence=copied_evidence,
                  expected_revision=None, request_sha256=digest, activation_authority=proof)
    _validate_knowledge_publication_replay(receipt, **kwargs)
    legacy = receipt.model_copy(update={"request_sha256": _knowledge_publication_v1_request_sha256(
        copied, chunks, expected_revision=None,
    )})
    if not items and proof is None:
        _validate_knowledge_publication_replay(legacy, **kwargs)
    else:
        try:
            _validate_knowledge_publication_replay(legacy, **kwargs)
        except api.KnowledgePublicationConflict as error:
            assert error.reason == "operation_mismatch"
        else:
            raise AssertionError("Legacy replay accepted weaker publication material")
    digests.append(digest)
assert len(set(digests)) == 3
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


def test_publication_contracts_preserve_aliases_stubs_and_annotations():
    import cayu.storage as storage
    import cayu.storage.memory as legacy
    from cayu.knowledge import publication_contracts

    for module_name, names in _OWNERS.items():
        owner = importlib.import_module("cayu.knowledge." + module_name)
        for name in names:
            canonical = getattr(owner, name)
            assert getattr(legacy, name) is canonical
            if name.startswith("_MAX_"):
                continue
            assert canonical.__module__ == owner.__name__
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
        for name in _EXPORTS:
            assert manifest[name] == (publication_contracts.__name__, name)
            assert imports[name] == publication_contracts.__name__
            assert getattr(package, name) is getattr(publication_contracts, name)


def test_retirement_contracts_compose_with_scope_snapshots_and_round_trip():
    from datetime import UTC, datetime

    from cayu.knowledge.activation_contracts import (
        _knowledge_activation_retirement,
        _knowledge_activation_retirement_json,
        _parse_knowledge_activation_retirement_json,
    )
    from cayu.knowledge.records import KnowledgeEntry
    from cayu.knowledge.scopes import (
        _knowledge_access_snapshot,
        _knowledge_access_snapshot_json,
        _parse_knowledge_access_snapshot_json,
    )

    now = datetime(2026, 1, 1, tzinfo=UTC)
    entry = KnowledgeEntry(id="entry", text="Exact material", labels={"team": "sales"})
    snapshot = _knowledge_access_snapshot(entry)
    retirement = _knowledge_activation_retirement(entry, retired_at=now)
    entry.labels["team"] = "changed"
    assert snapshot.labels == retirement.access_snapshot.labels == {"team": "sales"}
    assert snapshot.labels is not retirement.access_snapshot.labels
    assert (
        _parse_knowledge_access_snapshot_json(_knowledge_access_snapshot_json(snapshot)) == snapshot
    )
    assert (
        _parse_knowledge_activation_retirement_json(
            _knowledge_activation_retirement_json(retirement)
        )
        == retirement
    )
    for value in (snapshot, retirement):
        encoded = pickle.dumps(value)
        assert type(value).__module__.encode() in encoded
        restored = pickle.loads(encoded)
        assert type(restored) is type(value)
        assert restored == value
