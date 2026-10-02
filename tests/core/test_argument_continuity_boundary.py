"""Private argument persistence shares one contract below runtime."""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

import cayu
from cayu.messages import Message, ToolCallPart
from cayu.runtime import _argument_continuity as legacy
from cayu.sessions import _argument_continuity as canonical


def test_legacy_argument_continuity_imports_share_contracts_and_scope():
    for name in (
        "ArgumentContinuity",
        "append_record",
        "validate_records",
        "private_read_scope",
        "require_private_key_access",
        "_reading",
    ):
        assert getattr(legacy, name) is getattr(canonical, name)
    for name in (
        "STORAGE_KEY",
        "MAX_ROUNDS",
        "MAX_ROUND_BYTES",
        "MAX_CALL_BYTES",
        "MAX_RECORD_BYTES",
    ):
        assert getattr(legacy, name) == getattr(canonical, name)


def test_old_pickled_argument_batch_uses_the_canonical_contract():
    batch = canonical.ArgumentContinuity(
        nonce="a" * 32, profile="b" * 64, arguments={"call": {"text": "retained input"}}
    )
    current = pickle.dumps(batch, protocol=0)
    previous = current.replace(canonical.__name__.encode(), legacy.__name__.encode())
    assert previous != current
    restored = pickle.loads(previous)
    assert type(restored) is canonical.ArgumentContinuity
    assert restored.model_dump(mode="json") == batch.model_dump(mode="json")


@pytest.mark.parametrize("owner", (canonical, legacy), ids=("canonical", "legacy"))
def test_private_read_scope_is_shared_and_restored_after_nested_failure(owner):
    observer = legacy if owner is canonical else canonical
    key = canonical.STORAGE_KEY
    with pytest.raises(ValueError, match="runtime-owned"):
        observer.require_private_key_access(key, read=True)
    with owner.private_read_scope():
        observer.require_private_key_access(key, read=True)
        with (
            pytest.raises(RuntimeError, match="abort read"),
            observer.private_read_scope(),
        ):
            owner.require_private_key_access(key, read=True)
            with pytest.raises(ValueError, match="runtime-owned"):
                observer.require_private_key_access(key, read=False)
            raise RuntimeError("abort read")
        observer.require_private_key_access(key, read=True)
    with pytest.raises(ValueError, match="runtime-owned"):
        observer.require_private_key_access(key, read=True)


def test_private_records_operate_without_runtime_or_store_implementations():
    script = """
import importlib.abc
import json
import sys
from types import SimpleNamespace

class RejectExecutionAndStores(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname == "cayu.runtime" or fullname.startswith("cayu.runtime.")
                or fullname == "cayu.storage" or fullname.startswith("cayu.storage.")
                or fullname == "cayu.sessions.base"):
            raise AssertionError(f"Private argument persistence imported {fullname}")

sys.meta_path.insert(0, RejectExecutionAndStores())
from cayu.messages import Message
from cayu.sessions import _argument_continuity as owner
data = json.load(sys.stdin)
batch = owner.ArgumentContinuity.model_validate(data["batch"])
message = Message.model_validate(data["message"])
record = owner.append_record(
    None, continuity=batch, session=SimpleNamespace(id="session", instance_id="instance"),
    messages=[message], request_digest="d" * 64,
)
validated = owner.validate_records(record)
assert len(validated) == 1
assert validated[0][1] == batch
with owner.private_read_scope():
    owner.require_private_key_access(owner.STORAGE_KEY, read=True)
assert not owner._reading.get()
assert not any(name == "cayu.runtime" or name.startswith("cayu.runtime.")
               or name == "cayu.storage" or name.startswith("cayu.storage.")
               or name == "cayu.sessions.base" for name in sys.modules)
"""
    batch = canonical.ArgumentContinuity(
        nonce="a" * 32, profile="b" * 64, arguments={"call": {"text": "retained input"}}
    )
    message = Message(
        role="assistant",
        content=[
            ToolCallPart(
                tool_call_id="call",
                tool_name="private_tool",
                arguments={},
                tool_round_id="tround_" + "1" * 32,
                model_step_id="mstep_" + "2" * 32,
                model_attempt_id="matt_" + "3" * 32,
            )
        ],
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        input=json.dumps(
            {"batch": batch.model_dump(mode="json"), "message": message.model_dump(mode="json")}
        ),
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
