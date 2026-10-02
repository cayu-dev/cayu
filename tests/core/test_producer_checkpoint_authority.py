"""Producer checkpoint authority is shared below runtime execution."""

from __future__ import annotations

import asyncio
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

import cayu
from cayu.runtime import _producer_cleanup_contract as legacy_cleanup
from cayu.runtime import _producer_output_store as legacy
from cayu.sessions import _producer_checkpoint as canonical
from cayu.sessions import _producer_cleanup_contract as cleanup


def _index(session_id="session"):
    return canonical.NativeProducerIndex(
        session_id=session_id,
        session_instance_id="instance",
        operation_key=canonical.OPERATION_PREFIX + session_id,
        record_commitment="sha256:" + "a" * 64,
    )


def test_legacy_producer_imports_share_contracts_and_authority():
    for name in (
        "NativeProducerAttachment",
        "NativeProducerInvocation",
        "NativeProducerPausedStop",
        "NativeProducerIndex",
        "attachment_operation_key",
        "attachment_index",
        "_PUBLICATION",
        "_publication_scope",
        "checkpoint_visible",
        "require_operation_key_access",
        "project_checkpoint_root",
    ):
        assert getattr(legacy, name) is getattr(canonical, name)
    assert legacy.ROOT_KEY == canonical.ROOT_KEY
    assert legacy.OPERATION_PREFIX == canonical.OPERATION_PREFIX
    assert legacy_cleanup.NativeProducerCleanupReceipt is cleanup.NativeProducerCleanupReceipt
    assert legacy.NativeProducerCleanupReceipt is cleanup.NativeProducerCleanupReceipt


@pytest.mark.parametrize(
    "name",
    (
        "NativeProducerAttachment",
        "NativeProducerInvocation",
        "NativeProducerPausedStop",
        "NativeProducerIndex",
        "NativeProducerCleanupReceipt",
    ),
)
def test_old_pickled_producer_classes_resolve_to_the_session_owner(name):
    owner = cleanup if name == "NativeProducerCleanupReceipt" else canonical
    previous_owner = legacy_cleanup if owner is cleanup else legacy
    model = getattr(owner, name)
    encoded = pickle.dumps(model, protocol=0)
    previous = encoded.replace(owner.__name__.encode(), previous_owner.__name__.encode())
    assert previous != encoded
    assert pickle.loads(previous) is model


@pytest.mark.parametrize("owner", (canonical, legacy), ids=("canonical", "legacy"))
def test_producer_scope_is_shared_and_restored_after_nested_failure(owner):
    observer = legacy if owner is canonical else canonical
    index, other = _index(), _index("other")
    assert not observer.checkpoint_visible(session_id=index.session_id)
    with pytest.raises(PermissionError):
        observer.require_operation_key_access(index.operation_key, read=False)
    with owner._publication_scope(index):
        assert observer.checkpoint_visible(session_id=index.session_id)
        assert not observer.checkpoint_visible(session_id=other.session_id)
        for suffix in ("", ":excluded", ":cleanup", ":output"):
            observer.require_operation_key_access(index.operation_key + suffix, read=False)
        with pytest.raises(PermissionError):
            observer.require_operation_key_access(other.operation_key, read=False)
        with pytest.raises(RuntimeError, match="abort"), observer._publication_scope(other):
            assert owner.checkpoint_visible(session_id=other.session_id)
            assert not owner.checkpoint_visible(session_id=index.session_id)
            raise RuntimeError("abort")
        assert observer._PUBLICATION.get() is index
    assert observer._PUBLICATION.get() is None


def test_producer_root_requires_exact_scope_and_preserves_unowned_authority():
    index = _index()
    current = {canonical.ROOT_KEY: index.model_dump(mode="json")}
    with pytest.raises(PermissionError, match="cannot create"):
        canonical.project_checkpoint_root(None, current, session_id=index.session_id)
    preserved = canonical.project_checkpoint_root(current, {}, session_id=index.session_id)
    assert preserved == current[canonical.ROOT_KEY]
    preserved["operation_key"] = "detached"
    assert current[canonical.ROOT_KEY]["operation_key"] == index.operation_key
    with pytest.raises(ValueError, match="another session"):
        canonical.project_checkpoint_root(current, {}, session_id="other")
    with pytest.raises(ValueError):
        canonical.project_checkpoint_root(
            {canonical.ROOT_KEY: {"session_id": index.session_id}}, {}, session_id=index.session_id
        )
    excluded = index.model_copy(
        update={"state": "excluded", "exclusion_commitment": "sha256:" + "b" * 64}
    )
    replacement = {canonical.ROOT_KEY: excluded.model_dump(mode="json")}
    with pytest.raises(ValueError):
        canonical.project_checkpoint_root(current, replacement, session_id=index.session_id)
    with legacy._publication_scope(excluded):
        assert (
            canonical.project_checkpoint_root(current, replacement, session_id=index.session_id)
            == replacement[canonical.ROOT_KEY]
        )
        forged = {canonical.ROOT_KEY: {**replacement[canonical.ROOT_KEY], "operation_key": "other"}}
        with pytest.raises(ValueError):
            canonical.project_checkpoint_root(current, forged, session_id=index.session_id)


@pytest.mark.anyio
async def test_producer_scope_is_task_local_and_restored_on_cancellation():
    entered = asyncio.Event()
    index = _index()

    async def publish():
        try:
            with legacy._publication_scope(index):
                entered.set()
                await asyncio.Event().wait()
        finally:
            assert canonical._PUBLICATION.get() is None

    task = asyncio.create_task(publish())
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        assert canonical._PUBLICATION.get() is None
        assert not canonical.checkpoint_visible(session_id=index.session_id)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert canonical._PUBLICATION.get() is None


def test_producer_checkpoint_rules_operate_without_runtime_or_stores():
    script = """
import importlib.abc
import json
import sys

class RejectExecutionAndStores(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname == "cayu.runtime" or fullname.startswith("cayu.runtime.")
                or fullname == "cayu.storage" or fullname.startswith("cayu.storage.")
                or fullname == "cayu.sessions.base"):
            raise AssertionError(f"Producer checkpoint authority imported {fullname}")

sys.meta_path.insert(0, RejectExecutionAndStores())
from cayu.sessions import _producer_checkpoint as owner
index = owner.NativeProducerIndex.model_validate(json.load(sys.stdin))
with owner._publication_scope(index):
    assert owner.checkpoint_visible(session_id=index.session_id)
    owner.require_operation_key_access(index.operation_key, read=False)
    result = owner.project_checkpoint_root(
        None, {owner.ROOT_KEY: index.model_dump(mode="json")}, session_id=index.session_id,
    )
assert result == index.model_dump(mode="json")
assert owner._PUBLICATION.get() is None
assert not any(name == "cayu.runtime" or name.startswith("cayu.runtime.")
               or name == "cayu.storage" or name.startswith("cayu.storage.")
               or name == "cayu.sessions.base" for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        input=_index().model_dump_json(),
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
