"""Independent activation policy composition and legacy contract compatibility."""

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
    "KnowledgeGovernanceMode",
    "KnowledgeGovernanceConfig",
    "KnowledgeActivationDisposition",
    "KnowledgeActivationSource",
    "KnowledgeActivationRequest",
    "KnowledgeActivationDecision",
    "KnowledgeActivationAuthority",
    "KnowledgeActivationReceipt",
    "KnowledgeActivationConflict",
    "KnowledgeReviewApproval",
)
_BOUNDS = (
    "MAX_KNOWLEDGE_ACTIVATION_ANNOTATION_BYTES",
    "MAX_KNOWLEDGE_ACTIVATION_CHUNKS",
    "MAX_KNOWLEDGE_ACTIVATION_EVIDENCE_RECORDS",
    "MAX_KNOWLEDGE_ACTIVATION_EVALUATOR_RESULT_BYTES",
    "MAX_KNOWLEDGE_ACTIVATION_REQUEST_BYTES",
    "MAX_KNOWLEDGE_ACTIVATION_RECEIPT_BYTES",
)
_HELPERS = (
    "_knowledge_activation_schema_version",
    "_knowledge_activation_revision",
    "copy_knowledge_activation_request",
    "copy_knowledge_activation_decision",
    "copy_knowledge_activation_authority",
    "copy_knowledge_activation_receipt",
    "copy_knowledge_review_approval",
    "_knowledge_activation_receipt_json",
    "prepare_knowledge_activation_request",
)
_SHARED = {
    "records": (
        "_copy_entry_chunks",
        "_copy_entry_evidence",
        "_next_knowledge_revision",
        "_knowledge_publication_operation_id",
    ),
    "scopes": ("knowledge_access_scope_sha256", "_knowledge_access_scope_sha256"),
}


@pytest.mark.parametrize(
    "module_name", ("cayu", "cayu.storage", "cayu.knowledge.activation_contracts")
)
def test_activation_policies_compose_without_storage_implementations(module_name):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import asyncio
import importlib
import sys
from datetime import UTC, datetime

from cayu.knowledge.governance import decide_knowledge_activation
from cayu.knowledge.records import KnowledgeChunk, KnowledgeEntry, KnowledgeEvidence
from cayu.knowledge.scopes import KnowledgeAccessScope, knowledge_access_scope_sha256

api = importlib.import_module(sys.argv[1])
now = datetime(2026, 1, 1, tzinfo=UTC)
entry = KnowledgeEntry(id="entry", text="Exact material", status="pending",
                       created_at=now, updated_at=now)
chunk = KnowledgeChunk(id="chunk", entry_id=entry.id, chunk_index=0, text=entry.text)
evidence = KnowledgeEvidence(id="evidence", entry_id=entry.id, chunk_id=chunk.id,
                             source_type="test", source_id="source", source_revision="1",
                             created_at=now)
scope = KnowledgeAccessScope.privileged()

class Policy:
    def __init__(self, disposition):
        self.disposition = disposition

    async def decide_activation(self, request):
        return api.KnowledgeActivationDecision(
            request_sha256=request.fingerprint, disposition=self.disposition,
            policy_identity="application-policy", policy_version="1", code="tested",
            annotations={"rule": ["trusted-source"]},
        )

async def run():
    for mode in api.KnowledgeGovernanceMode:
        reviewed = mode is api.KnowledgeGovernanceMode.REVIEWED
        config = api.KnowledgeGovernanceConfig(mode=mode, **({} if reviewed else {
            "policy_identity": "application-policy", "policy_version": "1",
        }))
        request = api.prepare_knowledge_activation_request(
            entry, [chunk], evidence=[evidence], access_scope=scope, operation_id="operation",
            governance_mode=mode, source=api.KnowledgeActivationSource.CURATOR,
        )
        assert request.candidate_entry is not entry
        assert request.chunks[0] is not chunk and request.evidence[0] is not evidence
        assert request.access_scope is not scope
        assert request.access_scope_sha256 == knowledge_access_scope_sha256(scope)
        for disposition in api.KnowledgeActivationDisposition:
            if reviewed and disposition is not api.KnowledgeActivationDisposition.ROUTE_TO_REVIEW:
                continue
            authority = await decide_knowledge_activation(
                request, config=config, policy=None if reviewed else Policy(disposition),
            )
            assert type(authority) is api.KnowledgeActivationAuthority
            assert authority.request is not request
            assert authority.decision.disposition is disposition
            assert authority.decision.request_sha256 == request.fingerprint
            if disposition is not api.KnowledgeActivationDisposition.REJECT:
                receipt = api.KnowledgeActivationReceipt(
                    operation_id=request.operation_id, entry_id=entry.id, entry_revision=1,
                    expected_revision=None, publication_request_sha256="a" * 64,
                    authority=authority, committed_at=now,
                )
                assert receipt.authority.request.fingerprint == request.fingerprint
    assert not {
        "cayu.storage.memory", "cayu.storage.knowledge_sqlite", "cayu.storage.postgres",
    }.intersection(sys.modules)

asyncio.run(run())
""",
            module_name,
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_activation_contracts_preserve_aliases_stubs_and_runtime_annotations():
    import cayu.storage as storage
    import cayu.storage.memory as legacy
    from cayu.knowledge import activation_contracts as contracts

    owners = {name: contracts for name in (*_CLASSES, *_HELPERS, *_BOUNDS)}
    for module_name, names in _SHARED.items():
        module = importlib.import_module("cayu.knowledge." + module_name)
        owners.update(dict.fromkeys(names, module))
    for name, module in owners.items():
        canonical = getattr(module, name)
        assert getattr(legacy, name) is canonical
        if name in _BOUNDS:
            continue
        assert canonical.__module__ == module.__name__
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
        public_names = (*_CLASSES, *_BOUNDS, "prepare_knowledge_activation_request")
        if package is storage:
            public_names += ("knowledge_access_scope_sha256",)
        for name in public_names:
            owner = owners[name]
            assert manifest[name] == (owner.__name__, name)
            assert imports[name] == owner.__name__
            assert getattr(package, name) is getattr(owner, name)


def test_activation_values_round_trip_through_canonical_pickle_paths():
    from datetime import UTC, datetime

    from cayu.knowledge import activation_contracts as contracts
    from cayu.knowledge.records import KnowledgeChunk, KnowledgeEntry, KnowledgeStatus
    from cayu.knowledge.scopes import KnowledgeAccessScope

    now = datetime(2026, 1, 1, tzinfo=UTC)
    entry = KnowledgeEntry(
        id="entry", text="Reviewed knowledge", status="pending", created_at=now, updated_at=now
    )
    request = contracts.prepare_knowledge_activation_request(
        entry,
        [KnowledgeChunk(id="chunk", entry_id=entry.id, chunk_index=0, text=entry.text)],
        access_scope=KnowledgeAccessScope.privileged(),
        operation_id="approval",
        governance_mode=contracts.KnowledgeGovernanceMode.REVIEWED,
        source=contracts.KnowledgeActivationSource.REVIEW_APPROVAL,
        expected_revision=1,
    )
    decision = contracts.KnowledgeActivationDecision(
        request_sha256=request.fingerprint,
        disposition="activate",
        policy_identity="reviewer",
        policy_version="1",
        code="approved",
        annotations={"nested": ["reviewed"]},
    )
    authority = contracts.KnowledgeActivationAuthority(request=request, decision=decision)
    receipt = contracts.KnowledgeActivationReceipt(
        operation_id=request.operation_id,
        entry_id=entry.id,
        entry_revision=2,
        expected_revision=1,
        publication_request_sha256="a" * 64,
        authority=authority,
        committed_at=now,
    )
    approval = contracts.KnowledgeReviewApproval(
        entry=entry.model_copy(update={"revision": 2, "status": KnowledgeStatus.ACTIVE}),
        receipt=receipt,
    )
    copied = contracts.copy_knowledge_review_approval(approval)
    assert copied == approval and copied is not approval
    assert copied.entry is not approval.entry
    assert copied.receipt.authority.decision.annotations is not decision.annotations
    for value in (
        contracts.KnowledgeGovernanceConfig(),
        request,
        decision,
        authority,
        receipt,
        approval,
        contracts.KnowledgeGovernanceMode.REVIEWED,
        contracts.KnowledgeActivationSource.REVIEW_APPROVAL,
        contracts.KnowledgeActivationDisposition.ACTIVATE,
    ):
        encoded = pickle.dumps(value)
        assert contracts.__name__.encode() in encoded
        restored = pickle.loads(encoded)
        assert type(restored) is type(value)
        assert restored == value
