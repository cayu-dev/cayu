"""Publication contracts and validation compose independently of concrete stores."""

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
        ("authority", "CheckpointRootFieldProjection"),
        ("authority", "CheckpointRootFieldGuard"),
        ("authority", "checkpoint_root_field_projection"),
        ("authority", "checkpoint_root_field_projection_from_storage"),
        ("_operation_records", "RuntimePublicationOperationRecordMutation"),
        ("_operation_records", "runtime_publication_operation_record_value_digest"),
        ("_checkpoint_mutations", "runtime_publication_checkpoint_mutation"),
        ("_runtime_publication", "RuntimePublicationEventReference"),
        ("_runtime_publication", "RuntimePublicationRequest"),
        ("_runtime_publication", "RuntimePublicationReceipt"),
        ("_runtime_publication", "RuntimePublicationResult"),
        ("_runtime_publication", "runtime_publication_event_reference"),
        ("_runtime_publication_rules", "runtime_publication_request_digest"),
    ),
)
def test_publication_rules_preserve_public_identity(module, name):
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
            "cayu.storage.sqlite", "cayu.storage.postgres", "cayu.storage.memory",
            "cayu.sessions.base", "cayu.tasks.memory", "cayu.applications",
            "cayu.runtime._session_engine", "cayu.runtime._recovery_coordinator",
            "cayu.runtime._model_step_executor", "cayu.runtime._tool_round_executor",
        }:
            raise AssertionError(f"Publication rules imported {fullname}")
sys.meta_path.insert(0, RejectImplementations())

def rejected(call, error_type=ValueError):
    try:
        call()
    except error_type:
        return
    raise AssertionError("Invalid publication authority was accepted")
"""
            + code,
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_operation_record_mutations_preserve_cas_and_namespaces_without_stores():
    _without_stores(
        """
from cayu.sessions import _operation_records as rules
from cayu.sessions._checkpoint_mutations import SessionRuntimePublicationConflict

original = {"customer": {"values": [1]}}
replacement = {"values": [2]}
mutation = rules.RuntimePublicationOperationRecordMutation(key="customer",
    expected_value_digest=rules.runtime_publication_operation_record_value_digest(original["customer"]),
    value=replacement)
replacement["values"].clear()
result = rules._apply_runtime_publication_operation_record_mutations((mutation,), original)
assert result == {"customer": {"values": [2]}}
result["customer"]["values"].append(3)
assert original == {"customer": {"values": [1]}}
assert mutation.value == {"values": [2]}
rejected(lambda: rules._apply_runtime_publication_operation_record_mutations((mutation,), result),
    SessionRuntimePublicationConflict)
for key in (rules.RUNTIME_PUBLICATION_OPERATION_KEY_PREFIX + "receipt",
            rules.MODEL_COMPLETION_STAGE_OPERATION_KEY_PREFIX + "stage",
            rules.ZERO_WORK_INTERRUPTION_OPERATION_KEY):
    rejected(lambda: rules.RuntimePublicationOperationRecordMutation(
        key=key, expected_value_digest=None, value={}))
"""
    )


def test_checkpoint_projection_and_diff_preserve_bounded_authority_without_stores():
    _without_stores(
        """
from cayu.sessions import authority
from cayu.sessions._checkpoint_mutations import (
    apply_runtime_publication_checkpoint_mutation, runtime_publication_checkpoint_mutation,
)

source = {"same": [1], "remove": True, "change": 1}
target = {"same": [1], "new": {"nested": [2]}, "change": True}
mutation = runtime_publication_checkpoint_mutation(source, target)
assert [operation.key for operation in mutation.operations] == ["change", "new", "remove"]
target["new"]["nested"].clear()
result = apply_runtime_publication_checkpoint_mutation(mutation, source)
assert result == {"same": [1], "new": {"nested": [2]}, "change": True}
assert source == {"same": [1], "remove": True, "change": 1}
rejected(lambda: runtime_publication_checkpoint_mutation(source, None))
assert authority.checkpoint_root_field_projection(None, "value").is_present is False
assert authority.checkpoint_root_field_projection({"value": None}, "value").json_type == "null"
value = "1" * (authority.CHECKPOINT_ROOT_FIELD_SCALAR_MAX_CHARS + 1)
projected = authority.checkpoint_root_field_projection({"value": int(value)}, "value")
assert projected.scalar_text_truncated
assert len(projected.scalar_text) == authority.CHECKPOINT_ROOT_FIELD_SCALAR_MAX_CHARS
assert projected == authority.checkpoint_root_field_projection_from_storage(
    json_type="integer", scalar_text=value)
rejected(lambda: authority._validate_optional_runtime_publication_fence(True, "epoch"), TypeError)
"""
    )


def test_publication_contracts_detach_values_and_bind_referenced_events_without_stores():
    _without_stores(
        """
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.sessions import _runtime_publication as contracts
from cayu.sessions._checkpoint_mutations import RuntimePublicationMutation

event = Event(type=EventType.MODEL_COMPLETED, session_id="session", interaction_id="interaction",
    payload={"nested": [1]})
message = Message.text("assistant", "answer")
intent = {"nested": [1]}
request = contracts.RuntimePublicationRequest(publication_id="publication", kind="model-step",
    interaction_id="interaction", intent=intent, mutation=RuntimePublicationMutation(),
    transcript_messages=[message], events=[event])
event.payload["nested"].clear()
intent["nested"].clear()
assert request.intent == {"nested": [1]}
assert request.events[0].payload == {"nested": [1]}
assert request.transcript_messages[0] is not message
rejected(lambda: contracts.RuntimePublicationRequest(publication_id="duplicate", kind="model-step",
    intent={}, mutation=RuntimePublicationMutation(), transcript_messages=[], events=[event, event]))
reference = contracts.runtime_publication_event_reference(event)
contracts._validate_runtime_publication_event_references(
    (reference,), {event.id: event}, interaction_id="interaction")
event.payload["changed"] = True
rejected(lambda: contracts._validate_runtime_publication_event_references(
    (reference,), {event.id: event}, interaction_id="interaction"))
"""
    )


def test_publication_checkpoint_scope_restores_nested_and_cancelled_contexts_without_stores():
    _without_stores(
        """
import asyncio
from cayu.sessions import _checkpoint_publication as rules

def decode(session, checkpoint): return checkpoint
def encode(session, checkpoint): return checkpoint
def apply(session, checkpoint, mutation): return checkpoint

async def scenario():
    assert rules._active_runtime_publication_checkpoint_codec() is None
    with rules._runtime_publication_checkpoint_codec_scope(decode=decode, encode=encode, apply_mutation=apply):
        outer = rules._active_runtime_publication_checkpoint_codec()
        assert outer.decode is decode and outer.encode is encode and outer.apply_mutation is apply
        entered = asyncio.Event()
        async def child():
            assert rules._active_runtime_publication_checkpoint_codec() is outer
            try:
                with rules._runtime_publication_checkpoint_codec_scope(
                    decode=decode, encode=encode, apply_mutation=apply):
                    assert rules._active_runtime_publication_checkpoint_codec() is not outer
                    entered.set()
                    await asyncio.Future()
            finally:
                assert rules._active_runtime_publication_checkpoint_codec() is outer
        task = asyncio.create_task(child())
        await entered.wait()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("Cancellation disappeared")
        assert rules._active_runtime_publication_checkpoint_codec() is outer
    assert rules._active_runtime_publication_checkpoint_codec() is None
asyncio.run(scenario())
rejected(lambda: rules._validate_checkpoint_codec_pair(decode, None))
"""
    )


def test_publication_receipts_bind_exact_material_and_reject_tampering_without_stores():
    _without_stores(
        """
from datetime import UTC, datetime
from tests._session_provenance import session_fixture
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.sessions import _checkpoint_publication as projection, _runtime_publication_rules as rules
from cayu.sessions._checkpoint_mutations import RuntimePublicationMutation, SessionRuntimePublicationConflict
from cayu.sessions._runtime_publication import RuntimePublicationRequest
from cayu.sessions.records import SessionStatus

now = datetime(2026, 1, 1, tzinfo=UTC)
session = session_fixture(id="session", agent_name="agent", provider_name="fake", model="fake",
    status=SessionStatus.RUNNING, run_epoch=3, updated_at=now, last_activity_at=now)
event = Event(type=EventType.MODEL_COMPLETED, session_id=session.id, interaction_id="interaction",
    timestamp=now)
request = RuntimePublicationRequest(publication_id="publication", kind="model-step",
    interaction_id="interaction", intent={"step": 1}, mutation=RuntimePublicationMutation(),
    transcript_messages=[Message.text("assistant", "answer")], events=[event])
with projection._runtime_publication_checkpoint_codec_scope(
    decode=lambda s,c:c, encode=lambda s,c:c, apply_mutation=lambda s,c,m:c):
    codec = projection._active_runtime_publication_checkpoint_codec()
    prepared = rules._prepare_runtime_publication(session.id, request,
        expected_statuses={SessionStatus.RUNNING}, expected_run_epoch=3, expected_transcript_cursor=0)
assert prepared.checkpoint_codec is codec
assert projection._active_runtime_publication_checkpoint_codec() is None
assert prepared.request_digest == rules.runtime_publication_request_digest(session.id, request)
receipt = rules._build_runtime_publication_receipt(prepared, source_session=session,
    checkpoint=None, transcript_start_cursor=0, published_at=now)
record = rules._runtime_publication_receipt_record(receipt)
replayed = rules._reconstruct_runtime_publication_receipt(record, storage_key=prepared.storage_key,
    session_id=session.id, publication_id=request.publication_id, request_digest=prepared.request_digest)
assert replayed == receipt
rules._validate_runtime_publication_replay_receipt(replayed, prepared)
material = dict(transcript_messages=request.transcript_messages,
    transcript_interaction_ids=("interaction",), appended_events=request.events,
    durable_referenced_events=())
rules._validate_runtime_publication_durable_material(receipt, **material)
rejected(lambda: rules._validate_runtime_publication_durable_material(
    receipt, **{**material, "transcript_interaction_ids": (None,)}), SessionRuntimePublicationConflict)
rejected(lambda: rules._validate_runtime_publication_durable_material(
    receipt, **{**material, "appended_events": ()}), SessionRuntimePublicationConflict)
record["intent"]["step"] = 2
rejected(lambda: rules._reconstruct_runtime_publication_receipt(record,
    storage_key=prepared.storage_key, session_id=session.id, publication_id=request.publication_id),
    SessionRuntimePublicationConflict)
assert receipt.intent == {"step": 1}
"""
    )


def test_workspace_publication_validation_does_not_import_store_implementation():
    _without_stores(
        """
from cayu.sessions._checkpoint_mutations import RuntimePublicationMutation
from cayu.sessions._runtime_publication import RuntimePublicationRequest
from cayu.sessions._workspace_observation_publication import _validate_workspace_observation_publication

request = RuntimePublicationRequest(publication_id="workspace", kind="workspace-observation",
    intent={}, mutation=RuntimePublicationMutation(), transcript_messages=[], events=[])
try:
    _validate_workspace_observation_publication(request, session_id="session")
except ValueError as exc:
    assert str(exc) == "Workspace-observation publication intent is malformed."
else:
    raise AssertionError("Malformed workspace publication was accepted")
"""
    )


def test_tool_publication_binds_results_lifecycle_and_pending_marker_without_stores():
    _without_stores(
        """
from cayu.approvals.tools import PendingToolCallApproval
from cayu.events import Event, EventType
from cayu.messages import Message, ToolResultPart
from cayu.sessions import _tool_round_publication_rules as rules
from cayu.sessions._checkpoint_mutations import (
    RuntimePublicationCheckpointOperation, RuntimePublicationMutation, SessionRuntimePublicationConflict,
    runtime_publication_checkpoint_value_digest,
)
from cayu.sessions._pending_tool_round import PendingToolRound, pending_tool_round_identity
from cayu.sessions._runtime_publication import RuntimePublicationRequest, runtime_publication_event_reference

pending = PendingToolRound(model_step_id="mstep_" + "1" * 32, model_attempt_id="matt_" + "2" * 32,
    tool_round_id="tround_" + "3" * 32, agent_name="agent", environment_name=None, task_id=None,
    tool_calls=[PendingToolCallApproval(tool_call_id="call", tool_name="lookup", arguments={})])
identity = pending_tool_round_identity(pending)
part = ToolResultPart(tool_call_id="call", tool_name="lookup", content="answer", **identity.payload())
event = Event(type=EventType.TOOL_CALL_COMPLETED, session_id="session", interaction_id="interaction",
    tool_name="lookup", payload={**identity.payload(), "tool_call_id": "call", "idempotency_key": "stable",
    "result": {"content": part.content, "structured": part.structured,
               "artifacts": part.artifacts, "is_error": part.is_error}})
marker = pending.model_dump(mode="json")
request = RuntimePublicationRequest(publication_id="tool-round:" + identity.tool_round_id,
    kind="tool-round", interaction_id="interaction",
    intent={**identity.payload(), "round_id": identity.tool_round_id, "tool_call_ids": ["call"]},
    mutation=RuntimePublicationMutation(operations=[RuntimePublicationCheckpointOperation(
        key="pending_tool_round", action="delete", value=None,
        expected_value_digest=runtime_publication_checkpoint_value_digest(marker))]),
    transcript_messages=[Message(role="tool", content=(part,))], events=[],
    referenced_events=[runtime_publication_event_reference(event)])
rules._validate_tool_round_publication(request, {event.id: event})
rules._validate_tool_round_checkpoint_mutation(request, {"pending_tool_round": marker})
changed = {**marker, "task_id": "changed"}
rejected(lambda: rules._validate_tool_round_checkpoint_mutation(
    request, {"pending_tool_round": changed}), SessionRuntimePublicationConflict)
event.payload["result"]["content"] = "forged"
rejected(lambda: rules._validate_tool_round_publication(request, {event.id: event}))
"""
    )
