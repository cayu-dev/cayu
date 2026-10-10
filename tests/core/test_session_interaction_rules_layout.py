"""Shared interaction rules compose without concrete stores or runtime engines."""

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
        ("_checkpoint_mutations", "SessionRuntimePublicationConflict"),
        ("_checkpoint_mutations", "RuntimePublicationCheckpointOperation"),
        ("_checkpoint_mutations", "RuntimePublicationMutation"),
        ("_checkpoint_mutations", "runtime_publication_checkpoint_value_digest"),
        ("_checkpoint_mutations", "apply_runtime_publication_checkpoint_mutation"),
        ("_model_completion_publication", "ModelCompletionStageDisposition"),
        ("_model_completion_publication", "ModelCompletionStageSettlementRequest"),
    ),
)
def test_interaction_prerequisites_preserve_public_identity(module, name):
    canonical = getattr(importlib.import_module(f"cayu.sessions.{module}"), name)
    assert getattr(importlib.import_module("cayu.sessions.base"), name) is canonical
    assert pickle.loads(f"ccayu.sessions.base\n{name}\n.".encode()) is canonical
    assert pickle.loads(pickle.dumps(canonical)) is canonical


def _without_implementations(code: str) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys

class RejectImplementations(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("cayu.storage") or fullname in {
            "cayu.sessions.base", "cayu.tasks.memory", "cayu.applications",
            "cayu.runtime._session_engine", "cayu.runtime._recovery_coordinator",
            "cayu.runtime._model_step_executor", "cayu.runtime._tool_round_executor",
        }:
            raise AssertionError(f"Shared interaction rules imported {fullname}")
sys.meta_path.insert(0, RejectImplementations())

def rejected(call, error_type=ValueError):
    try:
        call()
    except error_type:
        return
    raise AssertionError("Invalid authority was accepted")
"""
            + code,
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_checkpoint_mutations_preserve_cas_and_detachment_without_implementations():
    _without_implementations(
        """
from cayu.sessions import _checkpoint_mutations as rules
from cayu.sessions.checkpoints import COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY

source = {"customer": {"values": [1]}}
replacement = {"values": [2]}
digest = rules.runtime_publication_checkpoint_value_digest(source["customer"])
operation = rules.RuntimePublicationCheckpointOperation(key="customer",
    expected_value_digest=digest, action="set", value=replacement)
mutation = rules.RuntimePublicationMutation(operations=[operation])
replacement["values"].clear()
result = rules.apply_runtime_publication_checkpoint_mutation(mutation, source)
assert result == {"customer": {"values": [2]}}
result["customer"]["values"].append(3)
assert source == {"customer": {"values": [1]}}
assert mutation.operations[0].value == {"values": [2]}
rejected(lambda: rules.apply_runtime_publication_checkpoint_mutation(mutation, result),
         rules.SessionRuntimePublicationConflict)
rejected(lambda: rules.RuntimePublicationMutation(operations=[operation, operation]))
rejected(lambda: rules.RuntimePublicationCheckpointOperation(
    key=COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY,
    expected_value_digest=None, action="set", value={}))
rejected(lambda: rules.RuntimePublicationCheckpointOperation(
    key="customer", expected_value_digest=None, action="delete"))
"""
    )


def test_stage_settlement_contract_validates_exact_supersession_without_implementations():
    _without_implementations(
        """
from cayu.sessions._model_completion_publication import (
    ModelCompletionStageDisposition as Disposition, ModelCompletionStageSettlementRequest,
)

fields = dict(stage_id="stage", logical_step_id="step", model_attempt_id="attempt",
    interaction_id="interaction", request_fingerprint="a" * 64,
    provider_effect_id="effect", preparation_digest="b" * 64,
    settlement_run_epoch=1, settled_reservation_ids=("reservation",), reason_code="replaced")
request = ModelCompletionStageSettlementRequest(**fields,
    disposition=Disposition.SUPERSEDED, superseding_stage_id="successor")
assert request.stage_id == "stage" and request.superseding_stage_id == "successor"
rejected(lambda: ModelCompletionStageSettlementRequest(**fields,
    disposition=Disposition.SUPERSEDED, superseding_stage_id="stage"))
rejected(lambda: ModelCompletionStageSettlementRequest(**fields,
    disposition=Disposition.FAILED_BEFORE_PROVIDER_EFFECT, superseding_stage_id="successor"))
rejected(lambda: ModelCompletionStageSettlementRequest(**{**fields, "stage_id": " padded "},
    disposition=Disposition.FAILED_BEFORE_PROVIDER_EFFECT))
"""
    )


def test_recovery_claim_and_completion_marker_rules_without_implementations():
    _without_implementations(
        """
from datetime import UTC, datetime, timedelta
from cayu.sessions import _completion_finalization as completion, recovery
from cayu.sessions.authority import SessionRunFenced

now = datetime(2026, 1, 1, tzinfo=UTC)
marker = {"version": 1, "claim_id": "owner", "claimed_at": now.isoformat(),
          "claim_expires_at": (now + timedelta(seconds=30)).isoformat()}
checkpoint = {recovery._INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY: marker}
assert recovery._active_unexpired_incomplete_recovery_claim_id(checkpoint, now=now) == "owner"
recovery._require_live_incomplete_recovery_claim_for_run_epoch_transfer(checkpoint, now=now)
rejected(lambda: recovery._require_live_incomplete_recovery_claim_for_run_epoch_transfer(
    checkpoint, now=now + timedelta(seconds=30)), SessionRunFenced)
assert recovery._active_unexpired_incomplete_recovery_claim_id(
    checkpoint, now=now + timedelta(seconds=30)) is None
mutation = {"operations": [{"key": completion.PENDING_COMPLETION_FINALIZATION_CHECKPOINT_KEY,
    "expected_value_digest": None, "action": "set", "value": {}}]}
completion._validate_completion_finalization_mutation_shape(mutation)
mutation["operations"][0]["expected_value_digest"] = "a" * 64
rejected(lambda: completion._validate_completion_finalization_mutation_shape(mutation))
"""
    )
