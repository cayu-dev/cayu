"""Shared continuation persistence stays independent of runtime execution."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import pytest
from tests.core.test_session_continuation import _admitted_store, _preparation, _ticket
from tests.core.test_temporary_service_target import side_admission

import cayu
from cayu.sessions import _session_continuation_store as continuation_store
from cayu.sessions._session_continuation import (
    CONTINUATION_NAMESPACE_KEY,
    ContinuationConflict,
    ContinuationRecord,
    continuation_operation_key,
)
from cayu.sessions._session_continuation_scope import publication_scope
from cayu.sessions._temporary_continuation import TemporaryServiceRecord, temporary_service_key
from cayu.sessions._temporary_service_target import TemporaryServiceTarget, target_service_key

MODULES = (
    "_external_wait_records",
    "_session_continuation_store",
    "_temporary_continuation_store",
    "_temporary_service_target",
    "_side_service_preparation",
)


@pytest.mark.parametrize("first_module", MODULES)
def test_persistence_records_and_projection_work_without_runtime_or_backends(first_module):
    script = """
import importlib
import importlib.abc
import json
import pickle
import sys

class RejectRuntimeAndStores(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname == "cayu.runtime" or fullname.startswith("cayu.runtime.")
                or fullname == "cayu.storage" or fullname.startswith("cayu.storage.")
                or fullname == "cayu.sessions.base"):
            raise AssertionError(f"Shared persistence imported {fullname}")

sys.meta_path.insert(0, RejectRuntimeAndStores())
importlib.import_module(f"cayu.sessions.{sys.argv[1]}")
from cayu.sessions import _session_continuation_store as index
from cayu.sessions import _temporary_continuation_store as service
from cayu.sessions import _temporary_service_target as target
from cayu.sessions import _side_service_preparation as preparation
from cayu.sessions._session_continuation import ContinuationConflict, continuation_operation_key
from cayu.sessions._session_continuation_scope import publication_scope
from cayu.sessions._temporary_continuation import (
    TemporaryServicePreparation, TemporaryServiceRecord, temporary_service_key,
)

prepared = preparation.prepare_selection(
    TemporaryServicePreparation.model_validate(json.load(sys.stdin))
)
intent = prepared.dispatch.intent
assert preparation.preparation_record_keys(prepared) == {
    intent.ticket.session_id: (
        continuation_operation_key(intent.ticket), temporary_service_key(intent.operation),
    ),
    intent.target.object_id: (target.target_service_key(intent.operation),),
}
receiving = target.TemporaryServiceTarget(
    service=TemporaryServiceRecord(admission=prepared, state="prepared")
)
excluded = target.exclude_side_target(receiving, prepared)
assert target.exclude_side_target(excluded, prepared) == excluded
acknowledged = target.acknowledge_side_target(excluded, excluded.service)
assert acknowledged.source_acknowledged
root = index.ContinuationRoot(namespace=intent.ticket.namespace)
checkpoint = {index.ROOT_KEY: root.model_dump(mode="json")}
assert index.project_checkpoint_root(
    checkpoint, {}, session_id=intent.ticket.session_id
) == checkpoint[index.ROOT_KEY]
try:
    index.project_checkpoint_root(None, checkpoint, session_id=intent.ticket.session_id)
except ContinuationConflict:
    pass
else:
    raise AssertionError("Generic checkpoint mutation granted continuation authority")
assert not index.checkpoint_visible()
with publication_scope(continuation_operation_key(intent.ticket)):
    assert index.checkpoint_visible()
    assert index.project_checkpoint_root(
        None, checkpoint, session_id=intent.ticket.session_id
    ) == checkpoint[index.ROOT_KEY]
assert not index.checkpoint_visible()
assert service.ContinuationRoot is index.ContinuationRoot
for value in (root, receiving, target.target_reference(receiving), acknowledged):
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        restored = pickle.loads(pickle.dumps(value, protocol=protocol))
        assert type(restored) is type(value)
        assert restored == value
        assert index.digest(restored.model_dump(mode="json")) == index.digest(
            value.model_dump(mode="json")
        )
"""
    result = subprocess.run(
        [sys.executable, "-c", script, first_module],
        input=side_admission().preparation.model_dump_json(),
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture(scope="module")
def operation_record_examples():
    async def prepare_record():
        store, session, interaction = await _admitted_store()
        ticket = _ticket(session, interaction)
        preparation = await _preparation(store, ticket)
        return ContinuationRecord(
            namespace=ticket.namespace, preparation=preparation, ticket=ticket
        )

    continuation = asyncio.run(prepare_record())
    service = TemporaryServiceRecord(admission=side_admission().preparation, state="prepared")
    target = TemporaryServiceTarget(service=service)
    key = continuation_operation_key(continuation.ticket)
    target_key = target_service_key(service.intent.operation)
    return {
        "namespace": (
            CONTINUATION_NAMESPACE_KEY,
            CONTINUATION_NAMESPACE_KEY,
            continuation.namespace.model_dump(mode="json"),
        ),
        "continuation": (key, key, continuation.model_dump(mode="json")),
        "service": (
            temporary_service_key(service.intent.operation),
            continuation_operation_key(service.intent.ticket),
            service.model_dump(mode="json"),
        ),
        "target": (target_key, target_key, target.model_dump(mode="json")),
    }


@pytest.fixture(params=("namespace", "continuation", "service", "target"))
def operation_record(request, operation_record_examples):
    return deepcopy(operation_record_examples[request.param])


def test_operation_record_requires_exact_publication_authority(operation_record):
    key, authority, record = operation_record
    original = deepcopy(record)
    for scope in (nullcontext(), publication_scope("session-continuation:unrelated")):
        with scope, pytest.raises(PermissionError, match="session owner"):
            continuation_store.require_operation_record_owner(key, record)
    with publication_scope(authority):
        assert continuation_store.require_operation_record_owner(key, record) is None
    with pytest.raises(PermissionError, match="session owner"):
        continuation_store.require_operation_record_owner(key, record)
    assert record == original


def test_operation_record_rejects_malformed_evidence_with_authority(operation_record):
    key, authority, _ = operation_record
    with publication_scope(authority), pytest.raises(ValueError):
        continuation_store.require_operation_record_owner(key, {})


def test_operation_record_requires_a_plain_object(operation_record):
    class RecordSubclass(dict):
        pass

    key, authority, record = operation_record
    for invalid in (None, [], RecordSubclass(record)):
        with (
            publication_scope(authority),
            pytest.raises(ContinuationConflict, match="not an object"),
        ):
            continuation_store.require_operation_record_owner(key, invalid)


@pytest.mark.parametrize("kind", ("continuation", "service", "target"))
def test_operation_record_key_must_match_its_identity(kind, operation_record_examples):
    key, authority, record = operation_record_examples[kind]
    changed_key = key[:-1] + ("0" if key[-1] != "0" else "1")
    with (
        publication_scope(authority if kind == "service" else changed_key),
        pytest.raises(ContinuationConflict, match="key"),
    ):
        continuation_store.require_operation_record_owner(changed_key, record)


def test_service_record_requires_parent_continuation_authority(operation_record_examples):
    key, _, record = operation_record_examples["service"]
    with publication_scope(key), pytest.raises(PermissionError, match="session owner"):
        continuation_store.require_operation_record_owner(key, record)


def test_unreserved_operation_records_do_not_require_continuation_authority():
    assert continuation_store.require_operation_record_owner("application:custom", None) is None


def test_operation_record_validation_works_without_runtime_or_backends(operation_record):
    script = """
import importlib.abc
import json
import sys

class RejectRuntimeAndStores(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname == "cayu.runtime" or fullname.startswith("cayu.runtime.")
                or fullname == "cayu.storage" or fullname.startswith("cayu.storage.")
                or fullname == "cayu.sessions.base"):
            raise AssertionError(f"Continuation validation imported {fullname}")

sys.meta_path.insert(0, RejectRuntimeAndStores())
from cayu.sessions._session_continuation_scope import publication_scope
from cayu.sessions._session_continuation_store import require_operation_record_owner

key, authority, record = json.load(sys.stdin)
with publication_scope(authority):
    assert require_operation_record_owner(key, record) is None
try:
    require_operation_record_owner(key, record)
except PermissionError:
    pass
else:
    raise AssertionError("Publication authority escaped its scope")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        input=json.dumps(operation_record),
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_runtime_does_not_reexport_the_private_record_validator():
    from cayu.runtime import _session_continuation

    assert not hasattr(_session_continuation, "require_operation_record_owner")
