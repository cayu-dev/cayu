"""Verified-work contracts stay below their runtime orchestration owners."""

from __future__ import annotations

import ast
import importlib
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

import cayu


@pytest.mark.parametrize(
    "first_module",
    (
        "cayu.tasks.completion_verifier_profiles",
        "cayu.tasks._verified_work_policy",
        "cayu.tasks._verified_work_authority",
        "cayu.storage.tasks_sqlite",
        "cayu.storage._postgres_verified_work",
    ),
)
def test_verified_work_contract_import_order_does_not_load_orchestration(first_module):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib
import sys
from typing import get_origin, get_type_hints

importlib.import_module(sys.argv[1])
from cayu.tasks import _verified_work_authority, _verified_work_policy
from cayu.tasks import completion_verifier_profiles as profiles

assert get_type_hints(profiles.CompletionVerifierProfileRecord)["profile"] is profiles.CompletionVerifierExecutionProfile
assert get_type_hints(_verified_work_authority.require_completion_verifier_profile_integrity)["profile"] is profiles.CompletionVerifierProfileRecord
assert get_origin(get_type_hints(_verified_work_policy.plan_decision_application)["return"]) is tuple
assert not {
    "cayu.runtime._verified_work_policy",
    "cayu.runtime._verified_work_authority",
    "cayu.runtime.completion_verifier_profiles",
    "cayu.verification._completion_verifier_coordinator",
    "cayu.verification._completion_result_resolver_coordinator",
    "cayu.verification._completion_decision_application_coordinator",
    "cayu.verification._verified_completion",
}.intersection(sys.modules)
""",
            first_module,
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_verifier_profiles_preserve_public_imports_and_legacy_pickle_names():
    from cayu.runtime import completion_verifier_profiles as legacy
    from cayu.tasks import completion_verifier_profiles as profiles

    public = [importlib.import_module(name) for name in ("cayu", "cayu.runtime")]
    assert legacy.__all__ == profiles.__all__
    for name in profiles.__all__:
        value = getattr(profiles, name)
        assert getattr(legacy, name) is value
        if name in cayu.__all__:
            assert all(getattr(module, name) is value for module in public)
        if isinstance(value, type):
            assert value.__module__ == profiles.__name__
            assert (
                pickle.loads(f"ccayu.runtime.completion_verifier_profiles\n{name}\n.".encode())
                is value
            )


def test_verifier_profile_values_keep_fingerprints_when_pickled():
    from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
    from cayu.tasks.completion_verifier_profiles import (
        build_completion_verifier_execution_profile,
        copy_completion_verifier_execution_profile,
    )
    from cayu.tasks.contracts import CompletionVerifierRef

    profile = build_completion_verifier_execution_profile(
        verifier=CompletionVerifierRef(
            verifier_id="layout-verifier", version="1", configuration_fingerprint="1" * 64
        ),
        adapter_identity=ExecutionProfileBehaviorIdentity(
            name="layout-verifier", behavior_version="1", implementation_version="1"
        ),
    )
    restored = pickle.loads(pickle.dumps(profile))
    assert type(restored) is type(profile)
    assert restored.model_dump(mode="json") == profile.model_dump(mode="json")
    assert copy_completion_verifier_execution_profile(restored).fingerprint == profile.fingerprint


def test_task_and_storage_contracts_use_canonical_verified_work_imports():
    root = Path(cayu.__file__).resolve().parent
    legacy_names = {
        "_verified_work_policy",
        "_verified_work_authority",
        "completion_verifier_profiles",
    }
    for package in (root / "tasks", root / "storage"):
        for path in package.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module]
                    if node.module == "cayu.runtime":
                        modules.extend(f"{node.module}.{alias.name}" for alias in node.names)
                assert not any(
                    module == f"cayu.runtime.{name}" or module.startswith(f"cayu.runtime.{name}.")
                    for module in modules
                    for name in legacy_names
                ), f"{path.relative_to(root)}:{node.lineno}: import the task contract owner"
