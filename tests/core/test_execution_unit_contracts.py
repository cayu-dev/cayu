"""Execution-unit contracts are shared independently of runtime execution."""

from __future__ import annotations

import importlib
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import get_type_hints

import pytest

import cayu
from cayu import execution_units as identities


@pytest.mark.parametrize("first_import", ("cayu.execution_units", "cayu"))
def test_execution_unit_contracts_work_without_runtime_or_storage(first_import):
    script = """
import importlib
import importlib.abc
import sys

class RejectExecution(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'cayu.runtime', 'cayu.storage', 'cayu.sessions'} or fullname.startswith(
            ('cayu.runtime.', 'cayu.storage.', 'cayu.sessions.')
        ):
            raise AssertionError(f'Execution-unit contracts imported {fullname}')

sys.meta_path.insert(0, RejectExecution())
importlib.import_module(sys.argv[1])
import cayu
from cayu import execution_units as units

step = cayu.new_model_step_identity()
attempt = step.new_attempt()
round_ = attempt.new_tool_round()
assert type(step) is units.ModelStepIdentity
assert type(attempt) is cayu.ModelAttemptIdentity is units.ModelAttemptIdentity
assert type(round_) is cayu.ToolRoundIdentity is units.ToolRoundIdentity
assert round_.model_step_id == step.model_step_id
assert round_.model_attempt_id == attempt.model_attempt_id
assert units.copy_tool_round_identity(round_) == round_
assert units.copy_tool_round_identity(round_) is not round_
assert round_.matches_payload(round_.payload())
assert not round_.matches_payload({**round_.payload(), 'model_attempt_id': 'different'})
payload = {**round_.payload(), 'budget_limit_id': 'forged', 'reservation_id': 'forged',
           'tool_call_id': 'call', 'custom': {'keep': True}}
units.strip_runtime_owned_execution_identity(payload)
assert payload == {'tool_call_id': 'call', 'custom': {'keep': True}}
assert cayu.BudgetLimitIdentity(budget_limit_id='blim_' + 'a' * 64).payload() == {
    'budget_limit_id': 'blim_' + 'a' * 64
}
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
    "name",
    (
        "BudgetLimitIdentity",
        "ModelStepIdentity",
        "ModelAttemptIdentity",
        "ToolRoundIdentity",
        "copy_model_step_identity",
        "copy_model_attempt_identity",
        "copy_tool_round_identity",
        "new_model_step_identity",
        "RUNTIME_OWNED_EXECUTION_IDENTITY_FIELDS",
        "strip_runtime_owned_execution_identity",
    ),
)
def test_public_execution_unit_imports_share_canonical_objects(name):
    runtime_path = importlib.import_module("cayu.runtime.execution_units")
    expected = getattr(identities, name)
    assert getattr(runtime_path, name) is expected
    assert pickle.loads(f"ccayu.runtime.execution_units\n{name}\n.".encode()) is expected
    if name not in {
        "RUNTIME_OWNED_EXECUTION_IDENTITY_FIELDS",
        "strip_runtime_owned_execution_identity",
    }:
        assert getattr(cayu, name) is expected
        assert getattr(importlib.import_module("cayu.runtime"), name) is expected


def test_execution_unit_inheritance_and_forward_annotations_share_canonical_classes():
    assert identities.ModelAttemptIdentity.__bases__ == (identities.ModelStepIdentity,)
    assert identities.ToolRoundIdentity.__bases__ == (identities.ModelAttemptIdentity,)
    assert (
        get_type_hints(identities.ModelStepIdentity.new_attempt)["return"]
        is identities.ModelAttemptIdentity
    )
    assert (
        get_type_hints(identities.ModelAttemptIdentity.new_tool_round)["return"]
        is identities.ToolRoundIdentity
    )


@pytest.mark.parametrize(
    "factory",
    (
        identities.new_model_step_identity,
        lambda: identities.new_model_step_identity().new_attempt(),
        lambda: identities.new_model_step_identity().new_attempt().new_tool_round(),
        lambda: identities.BudgetLimitIdentity(budget_limit_id="blim_" + "a" * 64),
    ),
    ids=("step", "attempt", "round", "budget"),
)
def test_execution_unit_instance_round_trips_keep_type_and_payload(factory):
    value = factory()
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        restored = pickle.loads(pickle.dumps(value, protocol=protocol))
        assert type(restored) is type(value)
        assert restored == value
        assert restored.payload() == value.payload()
    assert type(value).model_validate_json(value.model_dump_json()) == value
