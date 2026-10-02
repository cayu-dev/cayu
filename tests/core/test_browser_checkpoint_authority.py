"""Browser checkpoint authority is shared across canonical and legacy owners."""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

import cayu
from cayu.runtime import _browser_control_checkpoint as legacy
from cayu.sessions import _browser_control_checkpoint as canonical
from cayu.sessions.checkpoints import BROWSER_CONTROLS_CHECKPOINT_KEY as KEY
from cayu.tools.browser_control import (
    BrowserControlCheckpoint,
    BrowserControlConflict,
    BrowserControlIdentity,
    BrowserControlRecord,
    BrowserOperatorPurpose,
    closed_browser_control_successor,
)


@pytest.fixture
def mutations():
    record = BrowserControlRecord(
        identity=BrowserControlIdentity(
            session_id="session",
            session_instance_id="instance",
            run_epoch=1,
            interaction_id="interaction",
            execution_profile_fingerprint="a" * 64,
            environment_name="browser",
            allocation_fingerprint="b" * 64,
            browser_session_id="browser",
            worker_instance_id="worker",
            operator_purpose=BrowserOperatorPurpose(
                code="login", expected_origins=("https://app.example",)
            ),
        )
    )
    initial = BrowserControlCheckpoint(records=(record,))
    closed = closed_browser_control_successor(record)
    assert closed is not None
    return (
        canonical.BrowserControlCheckpointMutation("session", None, initial),
        canonical.BrowserControlCloseCheckpointMutation(
            "session", initial, initial.replace_record(expected=record, desired=closed)
        ),
    )


def test_legacy_browser_checkpoint_aliases_share_types_and_scope_state():
    for name, value in vars(canonical).items():
        if not name.startswith("_") and getattr(value, "__module__", None) == canonical.__name__:
            assert getattr(legacy, name) is value
    assert legacy.BROWSER_CONTROL_OPERATION_PREFIX == canonical.BROWSER_CONTROL_OPERATION_PREFIX
    assert legacy._MUTATION is canonical._MUTATION
    assert legacy._READ_SESSION is canonical._READ_SESSION


@pytest.mark.parametrize("index", (0, 1), ids=("mutation", "close"))
def test_old_pickled_browser_mutations_retain_exact_authority(mutations, index):
    mutation = mutations[index]
    current = pickle.dumps(mutation, protocol=0)
    previous = current.replace(canonical.__name__.encode(), legacy.__name__.encode())
    assert previous != current
    restored = pickle.loads(previous)
    assert type(restored) is type(mutation)
    assert restored == mutation
    assert canonical.browser_control_receipt(restored) == canonical.browser_control_receipt(
        mutation
    )
    assert canonical.browser_control_receipt_key(restored) == canonical.browser_control_receipt_key(
        mutation
    )


@pytest.mark.parametrize("owner", (canonical, legacy), ids=("canonical", "legacy"))
def test_browser_read_scope_is_shared_and_restored_across_import_paths(owner):
    observer = legacy if owner is canonical else canonical
    assert not observer.browser_control_checkpoint_visible(session_id="session")
    with (
        pytest.raises(RuntimeError, match="abort read"),
        owner.browser_control_checkpoint_read_scope("session"),
    ):
        assert observer.browser_control_checkpoint_visible(session_id="session")
        assert not observer.browser_control_checkpoint_visible(session_id="other")
        with observer.browser_control_checkpoint_read_scope("other"):
            assert owner.browser_control_checkpoint_visible(session_id="other")
            assert not owner.browser_control_checkpoint_visible(session_id="session")
        assert observer.browser_control_checkpoint_visible(session_id="session")
        raise RuntimeError("abort read")
    assert not observer.browser_control_checkpoint_visible(session_id="session")


@pytest.mark.parametrize("owner", (canonical, legacy), ids=("canonical", "legacy"))
def test_browser_mutation_scope_is_shared_and_restored_across_import_paths(owner, mutations):
    observer = legacy if owner is canonical else canonical
    mutation = mutations[0]
    replacement = {KEY: mutation.desired.model_dump(mode="json")}
    key = observer.browser_control_receipt_key(mutation)
    receipt = observer.browser_control_receipt(mutation)
    with (
        pytest.raises(RuntimeError, match="abort publication"),
        owner.browser_control_checkpoint_mutation_scope(mutation),
    ):
        assert (
            observer.project_browser_control_checkpoint(None, replacement, session_id="session")
            == replacement[KEY]
        )
        observer.require_browser_control_operation_owner(key, receipt)
        with (
            pytest.raises(BrowserControlConflict, match="cannot be nested"),
            observer.browser_control_checkpoint_mutation_scope(mutation),
        ):
            pytest.fail("A second scope must not replace the first owner")
        observer.require_browser_control_operation_owner(key, receipt)
        raise RuntimeError("abort publication")
    assert not observer.browser_control_checkpoint_visible(session_id="session")
    assert (
        observer.project_browser_control_checkpoint(None, replacement, session_id="session") is None
    )
    with pytest.raises(BrowserControlConflict, match="exact runtime owner"):
        observer.require_browser_control_operation_owner(key, receipt)


def test_browser_checkpoint_rules_import_and_operate_without_runtime(mutations):
    script = """
import importlib.abc
import pickle
import sys

class RejectRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "cayu.runtime" or fullname.startswith("cayu.runtime."):
            raise AssertionError(f"Browser checkpoint authority imported {fullname}")

sys.meta_path.insert(0, RejectRuntime())
from cayu.sessions import _browser_control_checkpoint as owner
from cayu.sessions.checkpoints import BROWSER_CONTROLS_CHECKPOINT_KEY as KEY
mutation = pickle.loads(sys.stdin.buffer.read())
replacement = {KEY: mutation.desired.model_dump(mode="json")}
with owner.browser_control_checkpoint_mutation_scope(mutation):
    assert owner.project_browser_control_checkpoint(None, replacement, session_id="session") == replacement[KEY]
    owner.require_browser_control_operation_owner(owner.browser_control_receipt_key(mutation), owner.browser_control_receipt(mutation))
assert not owner.browser_control_checkpoint_visible(session_id="session")
assert not any(name == "cayu.runtime" or name.startswith("cayu.runtime.") for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        input=pickle.dumps(mutations[0]),
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, (result.stdout + result.stderr).decode()
