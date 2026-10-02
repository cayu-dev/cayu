"""Canonical exports for the knowledge package migration."""

import ast
import importlib
import inspect
import os
import pickle
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import get_type_hints

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

_RELATION_CLASSES = (
    "KnowledgeRelation",
    "KnowledgeRelationConflict",
    "KnowledgeRelationDirection",
    "KnowledgeRelationKind",
    "KnowledgeRelationPublicationReceipt",
    "KnowledgeRelationQuery",
    "KnowledgeRelationResult",
    "KnowledgeLineageCurrentness",
    "KnowledgeLineageLink",
    "KnowledgeLineageQuery",
    "KnowledgeLineageResult",
    "KnowledgeLineageRole",
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
    from cayu.knowledge import records, relations, scopes

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
        (relations, _RELATION_CLASSES),
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


@pytest.mark.parametrize("module_name", ("cayu", "cayu.storage", "cayu.knowledge.relations"))
def test_relation_contracts_do_not_load_storage_implementations(module_name):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib
import sys
from datetime import UTC, datetime

from cayu.knowledge.records import KnowledgeRevisionRef

contracts = importlib.import_module(sys.argv[1])
a = KnowledgeRevisionRef(entry_id="a", revision=1)
b = KnowledgeRevisionRef(entry_id="b", revision=1)
relation = contracts.KnowledgeRelation(
    id="relation", subject=b, object=a, kind="contradicts",
    created_at=datetime(2026, 1, 1, tzinfo=UTC),
)
operation, prepared, fingerprint = contracts.prepare_knowledge_relations(
    [relation], operation_id="operation",
)
assert prepared[0].subject == a
assert prepared[0].object == b
receipt = contracts.KnowledgeRelationPublicationReceipt(
    operation_id=operation, relation_ids=["relation"], request_sha256=fingerprint,
    committed_at=relation.created_at,
)
page = contracts.KnowledgeRelationResult(
    query=contracts.KnowledgeRelationQuery(reference=a), relations=prepared,
)
assert page.relations == prepared
lineage = contracts.KnowledgeLineageResult(
    query=contracts.KnowledgeLineageQuery(reference=a),
    reference_current=a, reference_status="active",
    links=[contracts.KnowledgeLineageLink(
        relation_id="relation", kind="contradicts", role="contradicts", counterpart=b,
        counterpart_current=b, counterpart_status="active", currentness="current",
        unresolved_contradiction=True, created_at=relation.created_at,
    )],
)
assert lineage.links[0].unresolved_contradiction
assert not {
    "cayu.storage.memory",
    "cayu.storage.knowledge_sqlite",
    "cayu.storage.postgres",
}.intersection(sys.modules)
""",
            module_name,
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_relation_contracts_preserve_type_hints_stubs_and_helper_identity():
    import cayu.storage as storage
    import cayu.storage.memory as legacy
    from cayu.knowledge import records, relations

    for name in _RELATION_CLASSES:
        cls = getattr(relations, name)
        get_type_hints(cls)
        for method in vars(cls).values():
            if isinstance(method, classmethod | staticmethod):
                method = method.__func__
            if inspect.isfunction(method):
                get_type_hints(method)

    for name in (
        "copy_knowledge_relation",
        "copy_knowledge_relation_query",
        "copy_knowledge_lineage_link",
        "copy_knowledge_lineage_query",
        "copy_knowledge_relation_publication_receipt",
        "prepare_knowledge_relations",
    ):
        function = getattr(relations, name)
        assert getattr(legacy, name) is function
        get_type_hints(function)

    for package in (cayu, storage):
        manifest = importlib.import_module(f"{package.__name__}._exports").EXPORTS
        stub = ast.parse(Path(package.__file__).with_suffix(".pyi").read_text())
        imports = {
            alias.asname or alias.name: node.module
            for node in stub.body
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        for name in (*_RELATION_CLASSES, "prepare_knowledge_relations"):
            assert manifest[name] == (relations.__name__, name)
            assert imports[name] == relations.__name__
            assert getattr(package, name) is getattr(relations, name)
        assert package.DEFAULT_KNOWLEDGE_LIMIT == records.DEFAULT_KNOWLEDGE_LIMIT == 10
        assert imports["DEFAULT_KNOWLEDGE_LIMIT"] == records.__name__


def test_relation_values_support_pickle_round_trips():
    from cayu.knowledge import relations
    from cayu.knowledge.records import KnowledgeRevisionRef

    now = datetime(2026, 1, 1, tzinfo=UTC)
    a = KnowledgeRevisionRef(entry_id="a", revision=1)
    b = KnowledgeRevisionRef(entry_id="b", revision=1)
    relation = relations.KnowledgeRelation(
        id="relation",
        subject=a,
        object=b,
        kind="derived_from",
        created_at=now,
    )
    operation, prepared, fingerprint = relations.prepare_knowledge_relations(
        [relation],
        operation_id="operation",
    )
    receipt = relations.KnowledgeRelationPublicationReceipt(
        operation_id=operation,
        relation_ids=["relation"],
        request_sha256=fingerprint,
        committed_at=now,
    )
    query = relations.KnowledgeRelationQuery(reference=a)
    lineage_query = relations.KnowledgeLineageQuery(reference=a)
    link = relations.KnowledgeLineageLink(
        relation_id="relation",
        kind="derived_from",
        role="derived_from",
        counterpart=b,
        counterpart_current=b,
        counterpart_status="active",
        currentness="current",
        created_at=now,
    )
    for value in (
        relation,
        receipt,
        query,
        relations.KnowledgeRelationResult(query=query, relations=prepared),
        lineage_query,
        link,
        relations.KnowledgeLineageResult(
            query=lineage_query,
            reference_current=a,
            reference_status="active",
            links=[link],
        ),
        relations.KnowledgeRelationKind.DERIVED_FROM,
        relations.KnowledgeRelationDirection.BOTH,
        relations.KnowledgeLineageRole.DERIVED_FROM,
        relations.KnowledgeLineageCurrentness.CURRENT,
    ):
        encoded = pickle.dumps(value)
        assert relations.__name__.encode() in encoded
        restored = pickle.loads(encoded)
        assert type(restored) is type(value)
        assert restored == value
