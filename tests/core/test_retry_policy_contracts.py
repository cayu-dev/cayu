"""Retry configuration composes independently of runtime retry execution."""

from __future__ import annotations

import importlib
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import get_args, get_type_hints

import pytest

import cayu
from cayu.providers import retry_policy as policies


@pytest.mark.parametrize("first_import", ("cayu", "cayu.providers.retry_policy"))
def test_retry_policy_works_without_runtime_sessions_or_storage(first_import):
    script = """
import importlib
import importlib.abc
import sys

class RejectExecution(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'cayu.runtime', 'cayu.sessions', 'cayu.storage'} or fullname.startswith(
            ('cayu.runtime.', 'cayu.sessions.', 'cayu.storage.')
        ):
            raise AssertionError(f'Retry configuration imported {fullname}')

sys.meta_path.insert(0, RejectExecution())
importlib.import_module(sys.argv[1])
import cayu
from cayu.providers.retry_policy import RetryPolicy, copy_retry_policy

policy = cayu.RetryPolicy(max_attempts=3, initial_delay_s=0.0, jitter_s=0.0)
assert type(policy) is RetryPolicy
assert cayu.copy_retry_policy is copy_retry_policy
assert copy_retry_policy(policy) is policy
assert copy_retry_policy(None) == RetryPolicy()
assert RetryPolicy.model_validate_json(policy.model_dump_json()) == policy
try:
    RetryPolicy(max_attempts=True)
except ValueError:
    pass
else:
    raise AssertionError('Retry policy accepted a boolean attempt count')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, first_import],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "name", ("RetryPolicy", "copy_retry_policy", "DEFAULT_RETRYABLE_STATUS_CODES")
)
def test_public_retry_policy_imports_share_canonical_objects(name):
    expected = getattr(policies, name)
    assert getattr(importlib.import_module("cayu.runtime.retry_policy"), name) is expected
    assert pickle.loads(f"ccayu.runtime.retry_policy\n{name}\n.".encode()) is expected
    if name != "DEFAULT_RETRYABLE_STATUS_CODES":
        assert getattr(cayu, name) is expected
        assert getattr(importlib.import_module("cayu.runtime"), name) is expected


@pytest.mark.parametrize(
    ("module", "name"),
    (
        ("cayu.configuration", "RunDefaults"),
        ("cayu.sessions._pending_tool_round", "PendingToolRound"),
        ("cayu.approvals.tools", "PendingToolApproval"),
        ("cayu.approvals.user_input", "PendingUserInput"),
    ),
)
def test_saved_configuration_uses_the_canonical_retry_policy(module, name):
    model = getattr(importlib.import_module(module), name)
    annotation = model.model_fields["retry_policy"].annotation
    assert annotation is policies.RetryPolicy or policies.RetryPolicy in get_args(annotation)


def test_retry_policy_serialization_and_annotations_keep_canonical_identity():
    policy = policies.RetryPolicy(max_attempts=7, retry_on_status_codes=(429, 503))
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        restored = pickle.loads(pickle.dumps(policy, protocol=protocol))
        assert type(restored) is policies.RetryPolicy
        assert restored == policy
        assert policies.copy_retry_policy(restored) is restored
    assert get_type_hints(policies.copy_retry_policy)["return"] is policies.RetryPolicy


@pytest.mark.parametrize("module", ("cayu.providers.retry_policy", "cayu.runtime.retry_policy"))
def test_pending_round_round_trip_preserves_retry_policy(module):
    from cayu.sessions._pending_tool_round import PendingToolRound

    policy = importlib.import_module(module).RetryPolicy(
        max_attempts=7, max_unknown_attempts=3, retry_on_status_codes=(429, 503)
    )
    record = PendingToolRound(
        model_step_id="mstep_" + "1" * 32,
        model_attempt_id="matt_" + "2" * 32,
        tool_round_id="tround_" + "3" * 32,
        agent_name="worker",
        tool_calls=[{"tool_call_id": "call-1", "tool_name": "echo", "arguments": {}}],
        retry_policy=policy,
    )
    assert record.retry_policy is policy
    restored = PendingToolRound.model_validate_json(record.model_dump_json())
    assert type(restored.retry_policy) is policies.RetryPolicy
    assert restored.retry_policy == policy
    assert restored.model_dump(mode="json") == record.model_dump(mode="json")
