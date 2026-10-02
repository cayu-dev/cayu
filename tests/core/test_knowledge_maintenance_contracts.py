"""Compatibility and independent use of reviewed maintenance contracts."""

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

_CLASSES = (
    "KnowledgeMaintenanceProposal",
    "KnowledgeMaintenanceDecision",
    "KnowledgeMaintenanceDecisionReceipt",
    "KnowledgeMaintenanceDecisionKind",
    "KnowledgeMaintenanceOutcome",
    "KnowledgeMaintenanceConflict",
    "KnowledgeMaintenanceStale",
)
_BOUNDS = (
    "MAX_KNOWLEDGE_MAINTENANCE_SOURCES",
    "MAX_KNOWLEDGE_MAINTENANCE_BYTES",
    "MAX_KNOWLEDGE_MAINTENANCE_TEXT_BYTES",
    "MAX_KNOWLEDGE_MAINTENANCE_METADATA_BYTES",
)
_HELPERS = (
    "copy_knowledge_maintenance_proposal",
    "copy_knowledge_maintenance_decision",
    "copy_knowledge_maintenance_decision_receipt",
    "prepare_knowledge_maintenance_decision",
    "_knowledge_maintenance_identity",
    "_validate_knowledge_maintenance_record",
    "_validate_knowledge_maintenance_replay",
)


def _fresh_process(script, module_name):
    result = subprocess.run(
        [sys.executable, "-c", script, module_name],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "module_name", ("cayu", "cayu.storage", "cayu.knowledge.maintenance_contracts")
)
def test_maintenance_contracts_work_without_storage_implementations(module_name):
    _fresh_process(
        """
import importlib
import sys
from datetime import UTC, datetime

from cayu.knowledge.records import KnowledgeRevisionRef
from cayu.knowledge.relations import KnowledgeRelation
from cayu.knowledge.scopes import KnowledgeAccessScope

api = importlib.import_module(sys.argv[1])
now = datetime(2026, 1, 1, tzinfo=UTC)
source = KnowledgeRevisionRef(entry_id="source", revision=1)
replacement = KnowledgeRevisionRef(entry_id="replacement", revision=1)
proposal = api.KnowledgeMaintenanceProposal(
    id="proposal", replacement=replacement, sources=[source],
    relations=[KnowledgeRelation(
        id="relation", subject=replacement.model_copy(update={"revision": 2}),
        object=source, kind="supersedes", policy_id="policy", created_at=now,
    )],
    access_scope=KnowledgeAccessScope.privileged(), policy_id="policy", created_at=now,
    rationale="Reviewed consolidation", evidence_summary="Exact source revision reviewed",
)
decision = api.KnowledgeMaintenanceDecision(
    operation_id="operation", proposal_id=proposal.id,
    proposal_fingerprint=proposal.fingerprint, kind="approve", reviewer_type="user",
    reviewer="reviewer", reason="Approved exact proposal", decided_at=now,
)
copied_proposal, copied_decision, digest = api.prepare_knowledge_maintenance_decision(
    proposal, decision,
)
assert copied_proposal == proposal and copied_proposal is not proposal
assert copied_decision == decision and copied_decision is not decision
receipt = api.KnowledgeMaintenanceDecisionReceipt(
    operation_id=decision.operation_id, proposal_id=proposal.id,
    proposal_fingerprint=proposal.fingerprint, request_sha256=digest, outcome="applied",
    replacement=replacement.model_copy(update={"revision": 2}),
    archived_revisions=[source.model_copy(update={"revision": 2})],
    relation_ids=["relation"], committed_at=now,
)
from cayu.knowledge.maintenance_contracts import (
    _validate_knowledge_maintenance_record, _validate_knowledge_maintenance_replay,
)
_validate_knowledge_maintenance_record(proposal, decision, receipt)
_validate_knowledge_maintenance_replay(
    proposal, decision, receipt, proposal=copied_proposal, decision=copied_decision,
    request_sha256=digest,
)
assert not {
    "cayu.storage.memory", "cayu.storage.knowledge_sqlite", "cayu.storage.postgres",
}.intersection(sys.modules)
""",
        module_name,
    )


@pytest.mark.parametrize("module_name", ("maintenance", "maintenance_planning"))
def test_maintenance_authoring_modules_do_not_load_storage_implementations(module_name):
    _fresh_process(
        """
import importlib
import sys

importlib.import_module("cayu.knowledge." + sys.argv[1])
assert not {
    "cayu.storage.memory", "cayu.storage.knowledge_sqlite", "cayu.storage.postgres",
}.intersection(sys.modules)
""",
        module_name,
    )


def test_maintenance_contracts_preserve_exports_stubs_and_runtime_annotations():
    import cayu.storage as storage
    import cayu.storage.memory as legacy
    from cayu.knowledge import maintenance_contracts as contracts

    for name in (*_CLASSES, *_HELPERS, *_BOUNDS):
        canonical = getattr(contracts, name)
        assert getattr(legacy, name) is canonical
        if name in _BOUNDS:
            continue
        assert canonical.__module__ == contracts.__name__
        get_type_hints(canonical)
        assert pickle.loads(f"ccayu.storage.memory\n{name}\n.".encode()) is canonical
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
        for name in (*_CLASSES, *_BOUNDS, "prepare_knowledge_maintenance_decision"):
            assert manifest[name] == (contracts.__name__, name)
            assert imports[name] == contracts.__name__
            assert getattr(package, name) is getattr(contracts, name)


def test_maintenance_values_round_trip_through_the_canonical_pickle_path():
    from tests.core.knowledge_maintenance_conformance import (
        maintenance_decision,
        maintenance_proposal,
    )

    from cayu.knowledge import maintenance_contracts as contracts

    proposal = maintenance_proposal("pickle")
    decision = maintenance_decision(
        proposal, operation_id="operation", kind=contracts.KnowledgeMaintenanceDecisionKind.REJECT
    )
    _, _, digest = contracts.prepare_knowledge_maintenance_decision(proposal, decision)
    receipt = contracts.KnowledgeMaintenanceDecisionReceipt(
        operation_id=decision.operation_id,
        proposal_id=proposal.id,
        proposal_fingerprint=proposal.fingerprint,
        request_sha256=digest,
        outcome=contracts.KnowledgeMaintenanceOutcome.REJECTED,
        committed_at=decision.decided_at,
    )
    for value in (
        proposal,
        decision,
        receipt,
        contracts.KnowledgeMaintenanceDecisionKind.REJECT,
        contracts.KnowledgeMaintenanceOutcome.REJECTED,
    ):
        encoded = pickle.dumps(value)
        assert contracts.__name__.encode() in encoded
        restored = pickle.loads(encoded)
        assert type(restored) is type(value)
        assert restored == value
