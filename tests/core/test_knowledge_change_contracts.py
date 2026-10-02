"""Independent change-feed use and compatibility with existing package imports."""

import ast
import importlib
import inspect
import os
import pickle
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import get_type_hints

import pytest

import cayu

_CLASSES = (
    "KnowledgeChangeKind",
    "KnowledgeChangeConsumerConflict",
    "KnowledgeChange",
    "KnowledgeChangeBatch",
    "KnowledgeChangeClaim",
    "KnowledgeChangeConsumerState",
)
_BOUNDS = ("MAX_KNOWLEDGE_CHANGE_LIMIT", "MAX_KNOWLEDGE_CHANGE_SEQUENCE")
_HELPERS = (
    "copy_knowledge_change",
    "copy_knowledge_change_claim",
    "copy_knowledge_change_consumer_state",
    "_knowledge_change_claim_sha256",
    "_validate_knowledge_change_limit",
    "_validate_knowledge_change_sequence",
    "_knowledge_change_lease_seconds",
    "_knowledge_change_identity",
    "_initialize_knowledge_change_consumer_state",
)


@pytest.mark.parametrize("module_name", ("cayu", "cayu.storage", "cayu.knowledge.changes"))
def test_change_feed_contracts_work_without_storage_implementations(module_name):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib
import sys
from datetime import UTC, datetime, timedelta

api = importlib.import_module(sys.argv[1])
from cayu.knowledge.changes import (
    _initialize_knowledge_change_consumer_state,
    _knowledge_change_claim_sha256,
    copy_knowledge_change_claim,
)

now = datetime(2026, 1, 1, tzinfo=UTC)
change = api.KnowledgeChange(
    id="change", sequence=5, kind="revision_appended", entry_id="entry",
    entry_revision=2, committed_at=now, operation_id="publication",
)
items = [change]
page = api.KnowledgeChangeBatch(
    changes=items, after_sequence=2, next_after_sequence=5,
    high_water_sequence=8, truncated=True, limit=1,
)
assert page.changes == items and page.changes is not items
assert page.changes[0] is not change
items.clear()
assert len(page.changes) == 1
assert page.next_after_sequence == change.sequence
claim = api.KnowledgeChangeClaim(
    consumer_id="consumer", worker_id="worker", claim_id="claim", change=change,
    attempt=1, claimed_at=now, lease_expires_at=now + timedelta(seconds=30),
)
copied = copy_knowledge_change_claim(claim)
assert copied == claim and copied is not claim and copied.change is not claim.change
assert _knowledge_change_claim_sha256(copied) == _knowledge_change_claim_sha256(claim)
assert _knowledge_change_claim_sha256(claim.model_copy(update={"claim_id": "new-claim"})) != (
    _knowledge_change_claim_sha256(claim)
)
state = _initialize_knowledge_change_consumer_state(
    None, consumer_id="consumer", access_scope_sha256="a" * 64,
    baseline_sequence=page.next_after_sequence, now=now,
)
assert type(state) is api.KnowledgeChangeConsumerState
assert state.cursor_sequence == 5 and state.pending_change_sequence is None
try:
    _initialize_knowledge_change_consumer_state(
        state, consumer_id="consumer", access_scope_sha256="b" * 64,
        baseline_sequence=5, now=now,
    )
except api.KnowledgeChangeConsumerConflict as error:
    assert error.reason == "access_scope_mismatch"
else:
    raise AssertionError("Consumer accepted a different access scope")
assert not {
    "cayu.storage.memory", "cayu.storage.knowledge_sqlite", "cayu.storage.postgres",
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


def test_change_contracts_preserve_exports_stubs_and_runtime_annotations():
    import cayu.storage as storage
    import cayu.storage.memory as legacy
    from cayu.knowledge import changes

    for name in (*_CLASSES, *_HELPERS, *_BOUNDS):
        canonical = getattr(changes, name)
        assert getattr(legacy, name) is canonical
        if name in _BOUNDS:
            continue
        assert canonical.__module__ == changes.__name__
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
        for name in (*_CLASSES, *_BOUNDS):
            assert manifest[name] == (changes.__name__, name)
            assert imports[name] == changes.__name__
            assert getattr(package, name) is getattr(changes, name)


def test_change_feed_values_round_trip_through_canonical_pickle_paths():
    from cayu.knowledge import changes

    now = datetime(2026, 1, 1, tzinfo=UTC)
    change = changes.KnowledgeChange(
        id="change",
        sequence=5,
        kind="relation_published",
        entry_id="entry",
        entry_revision=2,
        relation_id="relation",
        committed_at=now,
    )
    batch = changes.KnowledgeChangeBatch(
        changes=[change], after_sequence=2, next_after_sequence=8, high_water_sequence=8, limit=3
    )
    claim = changes.KnowledgeChangeClaim(
        consumer_id="consumer",
        worker_id="worker",
        claim_id="claim",
        change=change,
        attempt=2,
        claimed_at=now,
        lease_expires_at=now + timedelta(seconds=30),
    )
    state = changes.KnowledgeChangeConsumerState(
        consumer_id=claim.consumer_id,
        access_scope_sha256="a" * 64,
        cursor_sequence=2,
        pending_change_sequence=change.sequence,
        pending_claim_id=claim.claim_id,
        pending_worker_id=claim.worker_id,
        pending_attempt=claim.attempt,
        claimed_at=claim.claimed_at,
        lease_expires_at=claim.lease_expires_at,
        updated_at=now,
    )
    for value in (change, batch, claim, state, changes.KnowledgeChangeKind.RELATION_PUBLISHED):
        encoded = pickle.dumps(value)
        assert changes.__name__.encode() in encoded
        restored = pickle.loads(encoded)
        assert type(restored) is type(value)
        assert restored == value
