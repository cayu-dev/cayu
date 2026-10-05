"""Shared continuation persistence stays independent of runtime execution."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from tests.core.test_temporary_service_target import side_admission

import cayu

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
