"""Shared model recovery parts retain legacy identities and independent imports."""

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
    "module_name",
    (
        "_model_completion_contracts",
        "_model_completion_delivery",
        "_model_stream_events",
        "_model_tool_discovery",
        "_provider_operation_recovery_owner",
    ),
)
def test_model_recovery_parts_import_without_execution_controllers(module_name: str) -> None:
    script = """
import importlib
import importlib.abc
import sys

blocked = {
    "cayu.applications",
    "cayu.runtime._model_step_executor",
    "cayu.runtime._session_engine",
    "cayu.runtime._recovery_coordinator",
}

class RejectControllers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked:
            raise AssertionError(f"Recovery component imported {fullname}")

sys.meta_path.insert(0, RejectControllers())
importlib.import_module(sys.argv[1])
assert not blocked.intersection(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, f"cayu.runtime.{module_name}"],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("module_name", "names"),
    (
        (
            "_model_completion_contracts",
            (
                "HostedToolDiscoveryRecoveryAuthority",
                "ModelCompletionRecoveryContext",
                "ModelCompletionDispatch",
                "ModelCompletionPublicationRequest",
                "ModelCompletionPublicationResult",
                "ModelCompletionDispatchNotAuthorized",
                "model_completion_recovery_context_from_stage",
            ),
        ),
        ("_model_completion_delivery", ("ModelAttemptFailed",)),
        (
            "_model_stream_events",
            ("_ModelStreamBoundaryValue", "_AssistantStreamBoundaryValue"),
        ),
        (
            "_provider_operation_recovery_owner",
            (
                "_ProviderRecoveryRequiredPublicationFailureEvidence",
                "_ProviderOperationStreamStatusError",
            ),
        ),
    ),
)
def test_legacy_model_contract_imports_preserve_exact_identity(
    module_name: str, names: tuple[str, ...]
) -> None:
    canonical = importlib.import_module(f"cayu.runtime.{module_name}")
    legacy = importlib.import_module("cayu.runtime._model_step_executor")
    for name in names:
        value = getattr(canonical, name)
        assert getattr(legacy, name) is value
        assert pickle.loads(f"c{legacy.__name__}\n{name}\n.".encode()) is value


def test_legacy_pickled_recovery_context_keeps_frozen_run_semantics() -> None:
    from cayu.budgets.run_limits import RunLimits
    from cayu.runtime._model_completion_contracts import (
        HostedToolDiscoveryRecoveryAuthority,
        ModelCompletionRecoveryContext,
    )

    context = ModelCompletionRecoveryContext(
        interaction_id="interaction",
        execution_profile_fingerprint="a" * 64,
        task_id="task",
        request_metadata={"labels": ["recovered"]},
        hosted_tool_discovery=HostedToolDiscoveryRecoveryAuthority(
            projection_sha256="b" * 64,
            targeted_tool_name_sha256s=("c" * 64,),
        ),
        limits=RunLimits(max_tool_calls=3),
    )
    current_pickle = pickle.dumps(context, protocol=0)
    legacy_pickle = current_pickle.replace(
        b"cayu.runtime._model_completion_contracts",
        b"cayu.runtime._model_step_executor",
    )
    assert legacy_pickle != current_pickle
    restored = pickle.loads(legacy_pickle)
    assert type(restored) is ModelCompletionRecoveryContext
    assert type(restored.hosted_tool_discovery) is HostedToolDiscoveryRecoveryAuthority
    assert restored.model_dump(mode="json") == context.model_dump(mode="json")
    assert restored.max_steps == 16
    assert restored.limits.max_tool_calls == 3
    restored.request_metadata["labels"].append("detached")
    assert context.request_metadata == {"labels": ["recovered"]}
