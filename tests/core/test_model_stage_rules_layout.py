"""Model-stage records and rules compose without concrete session stores."""

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
        ("_model_stage", "ModelCompletionStageRequest"),
        ("_model_stage", "ModelCompletionStage"),
        ("_model_stage", "ModelCompletionStageDispatch"),
        ("_model_stage", "ModelCompletionStageSettlement"),
        ("_model_stage", "ModelCompletionStageRecoveryFence"),
        ("_model_stage", "ModelCompletionManualRecoveryRequest"),
        ("_operation_records", "ModelCompletionStageRelease"),
        ("_operation_records", "SessionOperationPublication"),
    ),
)
def test_model_stage_public_imports_and_historical_pickles(module, name):
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
            "cayu.sessions.base", "cayu.storage.sqlite", "cayu.storage.postgres",
            "cayu.storage.memory", "cayu.tasks.memory", "cayu.applications",
            "cayu.runtime._session_engine", "cayu.runtime._recovery_coordinator",
            "cayu.runtime._model_step_executor", "cayu.runtime._tool_round_executor",
        }:
            raise AssertionError(f"Model-stage rules imported {fullname}")
sys.meta_path.insert(0, RejectImplementations())

def rejected(call, error_type=ValueError):
    try:
        call()
    except error_type:
        return
    raise AssertionError("Invalid model-stage authority was accepted")
"""
            + code,
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_model_stage_values_detach_intent_and_preserve_operation_authority_without_stores():
    _without_stores(
        """
from cayu.sessions._model_stage import ModelCompletionStageRequest
from cayu.sessions._operation_records import (
    MODEL_COMPLETION_STAGE_OPERATION_KEY_PREFIX, ModelCompletionStageRelease,
    SessionOperationPublication,
)

intent = {"nested": [1]}
request = ModelCompletionStageRequest(stage_id="stage", logical_step_id="step",
    dispatch_ordinal=0, purpose="assistant-turn", intent=intent, reservation_ids=())
intent["nested"].clear()
assert request.intent == {"nested": [1]}
checkpoint = {"nested": [2]}
records = {"customer": {"nested": [3]}}
release = ModelCompletionStageRelease(stage_id="stage", preparation_digest="a" * 64)
publication = SessionOperationPublication(checkpoint=checkpoint, operation_records=records,
    model_completion_stage_release=release)
checkpoint["nested"].clear()
records["customer"]["nested"].clear()
assert publication.checkpoint == {"nested": [2]}
assert publication.operation_records == {"customer": {"nested": [3]}}
assert publication.model_completion_stage_release == release
rejected(lambda: SessionOperationPublication(checkpoint={},
    operation_records={MODEL_COMPLETION_STAGE_OPERATION_KEY_PREFIX + "forged": {}}))
rejected(lambda: ModelCompletionStageRelease(stage_id="stage", preparation_digest="invalid"))
"""
    )
