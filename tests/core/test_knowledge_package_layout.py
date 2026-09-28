"""Canonical exports for the knowledge package migration."""

import importlib
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

import cayu

_MODULES = (
    ("curator", "LearningSignal"),
    ("enrichment", "KnowledgeEnrichmentQueueConfig"),
    ("governance", "KnowledgeActivationPolicyError"),
    ("maintenance", "KnowledgeMaintenanceSignalKind"),
    ("maintenance_governance", "KnowledgeMaintenanceGovernanceDisposition"),
    ("maintenance_persistence", "KnowledgeMaintenanceProposalPublicationOutcome"),
    ("maintenance_planning", "KnowledgeMaintenancePlanner"),
    ("semantic_watch", "KnowledgeSemanticWatchConfig"),
)


@pytest.mark.parametrize(("module_name", "symbol"), _MODULES)
def test_root_exports_share_canonical_identity(module_name, symbol):
    canonical = importlib.import_module(f"cayu.knowledge.{module_name}")
    assert getattr(cayu, symbol) is getattr(canonical, symbol)


@pytest.mark.parametrize(
    ("record_module", "scope_module"),
    (
        ("cayu", "cayu"),
        ("cayu.storage", "cayu.storage"),
        ("cayu.knowledge.records", "cayu.knowledge.scopes"),
    ),
)
def test_record_contracts_do_not_load_storage_implementations(record_module, scope_module):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib
import sys
from typing import get_type_hints

records = importlib.import_module(sys.argv[1])
scopes = importlib.import_module(sys.argv[2])
entry = records.KnowledgeEntry(id="entry", text="content")
scope = scopes.KnowledgeAccessScope.for_namespace(entry.namespace)
assert scope.allowed_namespaces == ["default"]
assert get_type_hints(records.KnowledgeEntry)["visibility"] is records.KnowledgeVisibility
assert get_type_hints(scopes.KnowledgeAccessScope)["required_labels"] == dict[str, str]
assert not {
    "cayu.storage.memory",
    "cayu.storage.knowledge_sqlite",
    "cayu.storage.postgres",
}.intersection(sys.modules)
""",
            record_module,
            scope_module,
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_record_contracts_preserve_public_and_legacy_class_identity():
    import cayu.storage as storage
    import cayu.storage.memory as legacy
    from cayu.knowledge import records, scopes

    for owner, names in (
        (
            records,
            (
                "KnowledgeEntry",
                "KnowledgeChunk",
                "KnowledgeEvidence",
                "KnowledgeEvidenceResult",
                "KnowledgeRevisionRef",
                "KnowledgeStatus",
                "KnowledgeVisibility",
                "KnowledgeActorType",
                "KnowledgeEvidenceRole",
                "KnowledgeEvidenceDisposition",
                "KnowledgeRevisionConflict",
                "KnowledgeChunkConflict",
                "KnowledgeEvidenceConflict",
                "KnowledgeEntryReadLimitExceeded",
            ),
        ),
        (scopes, ("KnowledgeAccessScope", "KnowledgeAccessDenied")),
    ):
        for name in names:
            canonical = getattr(owner, name)
            assert (
                getattr(cayu, name) is getattr(storage, name) is getattr(legacy, name) is canonical
            )
            assert canonical.__module__ == owner.__name__
            # Protocol 0 GLOBAL records model the original persisted class path.
            assert pickle.loads(f"ccayu.storage.memory\n{name}\n.".encode()) is canonical


def test_record_values_support_pickle_round_trips():
    from cayu.knowledge.records import (
        KnowledgeChunk,
        KnowledgeEntry,
        KnowledgeEvidence,
        KnowledgeRevisionRef,
    )
    from cayu.knowledge.scopes import KnowledgeAccessScope

    for value in (
        KnowledgeEntry(id="entry", text="content", metadata={"nested": ["value"]}),
        KnowledgeChunk(id="chunk", entry_id="entry", text="content", chunk_index=0),
        KnowledgeEvidence(
            id="evidence",
            entry_id="entry",
            source_type="document",
            source_id="source",
            source_revision="1",
        ),
        KnowledgeRevisionRef(entry_id="entry", revision=1),
        KnowledgeAccessScope.for_namespace("default", required_labels={"project": "one"}),
    ):
        encoded = pickle.dumps(value)
        assert type(value).__module__.encode() in encoded
        restored = pickle.loads(encoded)
        assert type(restored) is type(value)
        assert restored.model_dump(mode="json") == value.model_dump(mode="json")
