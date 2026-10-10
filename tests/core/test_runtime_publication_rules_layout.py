"""Publication contracts and validation compose independently of concrete stores."""

from __future__ import annotations

import importlib
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

import cayu


@pytest.mark.parametrize(
    ("module", "name"),
    (
        ("authority", "CheckpointRootFieldProjection"),
        ("authority", "CheckpointRootFieldGuard"),
        ("authority", "checkpoint_root_field_projection"),
        ("authority", "checkpoint_root_field_projection_from_storage"),
        ("_operation_records", "RuntimePublicationOperationRecordMutation"),
        ("_operation_records", "runtime_publication_operation_record_value_digest"),
        ("_checkpoint_mutations", "runtime_publication_checkpoint_mutation"),
    ),
)
def test_publication_rules_preserve_public_identity(module, name):
    canonical = getattr(importlib.import_module(f"cayu.sessions.{module}"), name)
    assert getattr(importlib.import_module("cayu.sessions.base"), name) is canonical
    assert pickle.loads(f"ccayu.sessions.base\n{name}\n.".encode()) is canonical
    assert pickle.loads(pickle.dumps(canonical)) is canonical


def _without_stores(code: str) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys

class RejectImplementations(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {
            "cayu.storage.sqlite", "cayu.storage.postgres", "cayu.storage.memory",
            "cayu.sessions.base", "cayu.tasks.memory", "cayu.applications",
            "cayu.runtime._session_engine", "cayu.runtime._recovery_coordinator",
            "cayu.runtime._model_step_executor", "cayu.runtime._tool_round_executor",
        }:
            raise AssertionError(f"Publication rules imported {fullname}")
sys.meta_path.insert(0, RejectImplementations())

def rejected(call, error_type=ValueError):
    try:
        call()
    except error_type:
        return
    raise AssertionError("Invalid publication authority was accepted")
"""
            + code,
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_operation_record_mutations_preserve_cas_and_namespaces_without_stores():
    _without_stores(
        """
from cayu.sessions import _operation_records as rules
from cayu.sessions._checkpoint_mutations import SessionRuntimePublicationConflict

original = {"customer": {"values": [1]}}
replacement = {"values": [2]}
mutation = rules.RuntimePublicationOperationRecordMutation(key="customer",
    expected_value_digest=rules.runtime_publication_operation_record_value_digest(original["customer"]),
    value=replacement)
replacement["values"].clear()
result = rules._apply_runtime_publication_operation_record_mutations((mutation,), original)
assert result == {"customer": {"values": [2]}}
result["customer"]["values"].append(3)
assert original == {"customer": {"values": [1]}}
assert mutation.value == {"values": [2]}
rejected(lambda: rules._apply_runtime_publication_operation_record_mutations((mutation,), result),
    SessionRuntimePublicationConflict)
for key in (rules.RUNTIME_PUBLICATION_OPERATION_KEY_PREFIX + "receipt",
            rules.MODEL_COMPLETION_STAGE_OPERATION_KEY_PREFIX + "stage",
            rules.ZERO_WORK_INTERRUPTION_OPERATION_KEY):
    rejected(lambda: rules.RuntimePublicationOperationRecordMutation(
        key=key, expected_value_digest=None, value={}))
"""
    )


def test_checkpoint_projection_and_diff_preserve_bounded_authority_without_stores():
    _without_stores(
        """
from cayu.sessions import authority
from cayu.sessions._checkpoint_mutations import (
    apply_runtime_publication_checkpoint_mutation, runtime_publication_checkpoint_mutation,
)

source = {"same": [1], "remove": True, "change": 1}
target = {"same": [1], "new": {"nested": [2]}, "change": True}
mutation = runtime_publication_checkpoint_mutation(source, target)
assert [operation.key for operation in mutation.operations] == ["change", "new", "remove"]
target["new"]["nested"].clear()
result = apply_runtime_publication_checkpoint_mutation(mutation, source)
assert result == {"same": [1], "new": {"nested": [2]}, "change": True}
assert source == {"same": [1], "remove": True, "change": 1}
rejected(lambda: runtime_publication_checkpoint_mutation(source, None))
assert authority.checkpoint_root_field_projection(None, "value").is_present is False
assert authority.checkpoint_root_field_projection({"value": None}, "value").json_type == "null"
value = "1" * (authority.CHECKPOINT_ROOT_FIELD_SCALAR_MAX_CHARS + 1)
projected = authority.checkpoint_root_field_projection({"value": int(value)}, "value")
assert projected.scalar_text_truncated
assert len(projected.scalar_text) == authority.CHECKPOINT_ROOT_FIELD_SCALAR_MAX_CHARS
assert projected == authority.checkpoint_root_field_projection_from_storage(
    json_type="integer", scalar_text=value)
rejected(lambda: authority._validate_optional_runtime_publication_fence(True, "epoch"), TypeError)
"""
    )
