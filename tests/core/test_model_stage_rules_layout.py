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


_PREPARED_STAGE = """
from datetime import UTC, datetime
from tests._session_provenance import session_fixture
from cayu.sessions import _model_stage_rules as rules
from cayu.sessions._model_stage import ModelCompletionStageRequest, SessionModelCompletionStageConflict
from cayu.sessions.records import SessionStatus

now = datetime(2026, 1, 1, tzinfo=UTC)
session = session_fixture(id="session", agent_name="agent", provider_name="fake", model="fake",
    status=SessionStatus.RUNNING, run_epoch=3, updated_at=now, last_activity_at=now)
request = ModelCompletionStageRequest(stage_id="stage", logical_step_id="step",
    dispatch_ordinal=0, purpose="context-compaction", reservation_ids=(),
    intent={"model_attempt_id": "attempt", "interaction_id": "interaction",
        "request_fingerprint": "a" * 64})
prepared = rules._prepare_model_completion_stage(session.id, request,
    expected_statuses={SessionStatus.RUNNING}, expected_run_epoch=3, expected_transcript_cursor=0)
record = rules._model_completion_stage_preparation_record(prepared, source_session=session,
    prepared_at=now)
keys = dict(session_id=session.id, stage_id=request.stage_id,
    preparation_storage_key=prepared.preparation_storage_key,
    terminal_storage_key=prepared.terminal_storage_key)
stage = rules._reconstruct_model_completion_stage(record, None, **keys)
assert stage is not None and stage.state == "in_flight"
"""


def test_model_stage_preparation_dispatch_and_settlement_reconcile_without_stores():
    _without_stores(
        _PREPARED_STAGE
        + """
from cayu.sessions._model_completion_publication import ModelCompletionStageDisposition

dispatch_record = rules._model_completion_stage_dispatch_record(stage, dispatched_at=now)
dispatch = rules._reconstruct_model_completion_stage_dispatch(dispatch_record,
    session_id=session.id, stage_id=stage.stage_id,
    storage_key=rules._model_completion_stage_dispatch_storage_key(stage.stage_id))
rules._validate_model_completion_stage_dispatch(dispatch, stage)
settlement_request = rules.model_completion_stage_settlement_request(stage,
    interaction_id="interaction", disposition=ModelCompletionStageDisposition.FAILED_BEFORE_PROVIDER_EFFECT,
    reason_code="dispatch_not_started", execution_profile_fingerprint=None,
    settlement_run_epoch=3, settled_reservation_ids=())
settlement_record = rules._model_completion_stage_settlement_record(stage,
    request=settlement_request, settled_at=now)
settlement = rules._reconstruct_model_completion_stage_settlement(settlement_record,
    session_id=session.id, stage_id=stage.stage_id,
    storage_key=rules._model_completion_stage_settlement_storage_key(stage.stage_id))
assert settlement.preparation_digest == stage.preparation_digest
assert settlement.provider_effect_id == dispatch.provider_effect_id
settlement_record["reason_code"] = "changed"
rejected(lambda: rules._reconstruct_model_completion_stage_settlement(settlement_record,
    session_id=session.id, stage_id=stage.stage_id,
    storage_key=rules._model_completion_stage_settlement_storage_key(stage.stage_id)),
    SessionModelCompletionStageConflict)
record["intent"]["interaction_id"] = "changed"
rejected(lambda: rules._reconstruct_model_completion_stage(record, None, **keys),
    SessionModelCompletionStageConflict)
"""
    )


def test_model_stage_terminal_material_replays_and_rejects_tampering_without_stores():
    _without_stores(
        _PREPARED_STAGE
        + """
from cayu.events import Event, EventType
from cayu.sessions._checkpoint_mutations import RuntimePublicationMutation
from cayu.sessions._runtime_publication import RuntimePublicationRequest
from cayu.sessions import _model_stage_publication as publication_rules

event = Event(type=EventType.MODEL_COMPLETED, session_id=session.id, timestamp=now,
    interaction_id="interaction", payload={"purpose": "context_compaction"})
publication = RuntimePublicationRequest(publication_id=stage.logical_step_id,
    kind="context-compaction", interaction_id="interaction", intent=stage.intent,
    mutation=RuntimePublicationMutation(), transcript_messages=(), events=(event,))
publication_rules._validate_model_completion_stage_publication(publication,
    session_id=session.id, stage=stage)
terminal_prepared = rules._prepare_model_completion_stage_terminal(session.id,
    stage_id=stage.stage_id, publication=publication)
terminal = rules._model_completion_stage_terminal_record(terminal_prepared,
    stage=stage, completed_at=now)
completed = rules._reconstruct_model_completion_stage(record, terminal, **keys)
assert completed is not None and completed.publication == publication
assert completed.publication is not publication
rules._validate_model_completion_stage_terminal_replay(completed, terminal_prepared)
terminal["publication"]["events"][0]["payload"]["purpose"] = "changed"
rejected(lambda: rules._reconstruct_model_completion_stage(record, terminal, **keys),
    SessionModelCompletionStageConflict)
rejected(lambda: rules._reconstruct_model_completion_stage(None, terminal, **keys),
    SessionModelCompletionStageConflict)
"""
    )
