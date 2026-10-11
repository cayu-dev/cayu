"""Shared delivery and closure rules compose without concrete store owners."""

from __future__ import annotations

import importlib
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

import cayu


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
        if fullname in {
            "cayu.sessions.base", "cayu.sessions.store", "cayu.sessions.memory", "cayu.storage.sqlite", "cayu.storage.postgres",
            "cayu.storage.memory", "cayu.tasks.memory", "cayu.applications",
            "cayu.runtime._session_engine", "cayu.runtime.session_closure",
            "cayu.runtime._model_step_executor", "cayu.runtime._tool_round_executor",
        }:
            raise AssertionError(f"Shared delivery rules imported {fullname}")
sys.meta_path.insert(0, RejectImplementations())

def rejected(call, error_type=ValueError):
    try:
        call()
    except error_type:
        return
    raise AssertionError("Invalid delivery or closure authority was accepted")
"""
            + code,
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_completion_event_reservations_compose_without_stores():
    _without_implementations(
        """
from datetime import UTC, datetime, timedelta
from cayu.sessions import _checkpoint_preservation as rules

now = datetime(2026, 1, 1, tzinfo=UTC)
identity = dict(publication_id="completion-result-publication:v1:" + "a" * 64,
    authority_sha256="a" * 64, owner_id="completion-result-owner:v1:" + "b" * 64)
original = {"customer": {"items": [1]}}
reserved = rules._reserve_completion_result_event_publication(original, **identity,
    owner_expires_at=now + timedelta(seconds=10), now=now)
assert original == {"customer": {"items": [1]}}
assert rules._completion_result_event_publication_delete_block_reason(reserved, now=now)
renewed = rules._renew_completion_result_event_publication(reserved, **identity,
    owner_expires_at=now + timedelta(seconds=20), now=now + timedelta(seconds=5))
assert rules._completion_result_event_publication_delete_block_reason(
    reserved, now=now + timedelta(seconds=10)) is None
assert rules._completion_result_event_publication_delete_block_reason(
    renewed, now=now + timedelta(seconds=10))
for release in (rules._release_completion_result_event_publication,
                rules._complete_completion_result_event_publication):
    cleared = release(renewed, **identity, require_present=True, now=now)
    assert cleared == original
    rejected(lambda: release(cleared, **identity, require_present=True, now=now))
    rejected(lambda: release(renewed, **{**identity, "authority_sha256": "c" * 64},
        require_present=True, now=now))
rejected(lambda: rules._reserve_completion_result_event_publication(None, **identity,
    owner_expires_at=now, now=now))
"""
    )


def test_first_event_delivery_validates_exact_unclaimed_attempt_without_stores():
    _without_implementations(
        """
from cayu.sessions.event_delivery import (
    PersistedEventSideEffectDelivery, PersistedEventSideEffectStatus,
    _copy_pending_first_event_delivery, _copy_failed_first_delivery_retirement,
    _copy_optional_event_id,
)

pending = PersistedEventSideEffectDelivery(session_id="session", event_id="event",
    event_sequence=1, status=PersistedEventSideEffectStatus.PENDING)
copied = _copy_pending_first_event_delivery(pending)
assert copied == pending and copied is not pending
failed = pending.model_copy(update={"status": PersistedEventSideEffectStatus.FAILED, "attempts": 1})
assert _copy_failed_first_delivery_retirement(failed) == failed
rejected(lambda: _copy_pending_first_event_delivery(failed))
rejected(lambda: _copy_failed_first_delivery_retirement(pending))
rejected(lambda: _copy_failed_first_delivery_retirement(failed.model_copy(update={"attempts": 2})))
rejected(lambda: _copy_pending_first_event_delivery(pending.model_copy(update={"claim_id": "owned"})))
assert _copy_optional_event_id(None, "event") is None
assert _copy_optional_event_id("event", "event") == "event"
rejected(lambda: _copy_optional_event_id("x" * 10000, "event"))
"""
    )


@pytest.mark.parametrize(
    "name", ("QueuedDispatchTerminalReceipt", "QueuedDispatchTerminalReceiptQuery")
)
def test_queued_dispatch_public_types_keep_identity_and_historical_pickles(name):
    canonical = getattr(importlib.import_module("cayu.sessions._terminal_evidence"), name)
    assert getattr(importlib.import_module("cayu.sessions.base"), name) is canonical
    assert pickle.loads(f"ccayu.sessions.base\n{name}\n.".encode()) is canonical
    assert pickle.loads(pickle.dumps(canonical)) is canonical


def test_queued_run_receipts_transfer_and_release_exact_retention_without_stores():
    _without_implementations(
        """
from tests._session_provenance import session_fixture
from cayu.events import Event, EventType
from cayu.sessions import _terminal_evidence as rules
from cayu.sessions.authority import SessionRunFenced
from cayu.sessions.records import SessionStatus

session = session_fixture(id="session", agent_name="agent", provider_name="fake", model="fake",
    status=SessionStatus.PENDING, run_epoch=2)
operation_id = "a" * 64
event_id = rules.queued_dispatch_terminal_event_id(operation_id)
checkpoint = rules._checkpoint_with_session_run_operation(checkpoint=None,
    current_session=session, operation_id=operation_id, terminal_event_id=event_id,
    queue_task_id="task")
operation = rules._session_run_operation_from_checkpoint(checkpoint)
assert operation.run_epoch == 3
bound = rules._event_with_session_run_operation(
    Event(type=EventType.SESSION_COMPLETED, session_id=session.id), operation)
assert bound.id == event_id
assert bound.payload[rules.SESSION_RUN_OPERATION_ID_PAYLOAD_KEY] == operation_id
retained = rules._checkpoint_after_session_run_operation_cleanup(checkpoint,
    operation=operation, retain_terminal_receipt=True)
assert rules._session_run_operation_from_checkpoint(retained) is None
receipts = rules._queued_dispatch_terminal_receipts_for_session(session.id, retained,
    durable_event_ids={event_id})
assert len(receipts) == 1 and receipts[0].queue_task_id == "task"
rejected(lambda: rules._checkpoint_with_session_run_operation(checkpoint=retained,
    current_session=session, operation_id=operation_id, terminal_event_id=event_id,
    queue_task_id="task"), SessionRunFenced)
rejected(lambda: rules._checkpoint_after_queued_dispatch_acknowledgement(retained,
    queue_task_id="other", operation_id=operation_id, terminal_event_id=event_id), RuntimeError)
cleared = rules._checkpoint_after_queued_dispatch_acknowledgement(retained,
    queue_task_id="task", operation_id=operation_id, terminal_event_id=event_id)
assert cleared == {}
assert rules._checkpoint_after_queued_dispatch_acknowledgement(cleared,
    queue_task_id="task", operation_id=operation_id, terminal_event_id=event_id) is None
"""
    )


def test_transcript_and_queue_finalization_rules_keep_exact_authority_without_stores():
    _without_implementations(
        """
from tests._session_provenance import session_fixture
from cayu.sessions import _checkpoint_publication as transcript
from cayu.sessions import _completion_finalization as completion
from cayu.sessions.records import SessionStatus
from cayu.sessions.authority import SessionRunFenced

session = session_fixture(id="session", agent_name="agent", provider_name="fake", model="fake",
    status=SessionStatus.RUNNING, run_epoch=1)
checkpoint = transcript._initial_transcript_pending_checkpoint(session, "interaction")
rejected(lambda: transcript._checkpoint_after_initial_transcript_publication(checkpoint,
    interaction_id="different"), RuntimeError)
completed = transcript._checkpoint_after_initial_transcript_publication(checkpoint,
    interaction_id="interaction")
assert transcript._initial_transcript_pending_interaction_id(checkpoint) == "interaction"
assert transcript._initial_transcript_pending_interaction_id(completed) is None
key = completion.PENDING_COMPLETION_FINALIZATION_CHECKPOINT_KEY
marker = dict(version=1, outcome="completed", environment_name="environment",
    binding_generation_id="generation", execution_profile_fingerprint="profile", binding_state={})
raw = dict(operations=[dict(key=key, expected_value_digest=None, action="set", value=marker)])
prepared = completion._prepare_queue_completion_checkpoint_mutation(raw, SessionStatus.RUNNING)
marker["binding_state"]["changed"] = True
assert prepared.operations[0].value["binding_state"] == {}
rejected(lambda: completion._prepare_queue_completion_checkpoint_mutation(raw, SessionStatus.COMPLETED))
rejected(lambda: completion._apply_queue_completion_checkpoint_mutation(session, prepared, None),
    SessionRunFenced)
"""
    )


def test_child_lifecycle_requires_current_durable_occurrence_without_stores():
    _without_implementations(
        """
from tests._session_provenance import session_fixture
from cayu.events import Event, EventType
from cayu.sessions import _child_lifecycle_rules as rules
from cayu.sessions.records import EventRecord, SessionStatus

parent = session_fixture(id="parent", agent_name="agent", provider_name="fake", model="fake")
child = session_fixture(id="child", parent_session_id=parent.id, agent_name="agent",
    provider_name="fake", model="fake", status=SessionStatus.COMPLETED)
def entry(records):
    return rules._child_session_lifecycle_entry(parent=parent, child=child,
        records_by_type=records, consumption_record=None)
terminal = EventRecord(sequence=2, event=Event(type=EventType.SESSION_COMPLETED, session_id=child.id))
assert entry({}) is None
fresh = entry({EventType.SESSION_COMPLETED: terminal})
assert fresh.freshness.value == "fresh" and fresh.occurrence.source_id == terminal.event.id
resumed = EventRecord(sequence=3, event=Event(type=EventType.SESSION_RESUMED, session_id=child.id))
assert entry({EventType.SESSION_COMPLETED: terminal, EventType.SESSION_RESUMED: resumed}) is None
rejected(lambda: rules._child_session_lifecycle_entry(parent=parent,
    child=child.model_copy(update={"parent_session_id": "other"}),
    records_by_type={}, consumption_record=None), RuntimeError)
"""
    )


def test_child_consumption_replays_only_exact_occurrence_and_logical_step_without_stores():
    _without_implementations(
        """
from datetime import UTC, datetime
from tests._session_provenance import session_fixture
from cayu.events import Event, EventType
from cayu.runtime._child_session_notifications import CHILD_SESSION_NOTIFICATION_INTENT_KEY
from cayu.sessions import _child_lifecycle_rules as rules
from cayu.sessions._model_stage import ModelCompletionStage, SessionModelCompletionStageConflict
from cayu.sessions.records import EventRecord, SessionStatus

now = datetime(2026, 1, 1, tzinfo=UTC)
parent = session_fixture(id="parent", agent_name="agent", provider_name="fake", model="fake")
child = session_fixture(id="child", parent_session_id=parent.id, agent_name="agent",
    provider_name="fake", model="fake", status=SessionStatus.COMPLETED)
event = EventRecord(sequence=2, event=Event(type=EventType.SESSION_COMPLETED, session_id=child.id))
_, occurrence = rules._child_session_lifecycle_occurrence(child, {EventType.SESSION_COMPLETED: event})
binding = dict(parent_session_id=parent.id, parent_session_instance_id=parent.instance_id,
    claims=[dict(child_session_id=child.id, child_session_instance_id=child.instance_id,
                 occurrence=occurrence.model_dump(mode="json"))])
stage = ModelCompletionStage(session_id=parent.id, stage_id="stage", logical_step_id="step",
    dispatch_ordinal=0, purpose="assistant-turn", state="in_flight",
    intent={CHILD_SESSION_NOTIFICATION_INTENT_KEY: binding,
            "model_step_id": "step", "model_attempt_id": "attempt"}, reservation_ids=(),
    preparation_request_digest="a"*64, preparation_digest="b"*64,
    source_status=SessionStatus.RUNNING, source_run_epoch=1, source_transcript_cursor=0,
    prepared_at=now)
record = rules._child_session_notification_consumption_record(parent=parent, child=child,
    occurrence=occurrence, stage=stage, consumed_at=now)
raw = record.model_dump(mode="json")
retry = record.model_copy(update={"stage_id": "retry", "model_attempt_id": "retry"})
assert rules._child_session_notification_consumption_replays(raw, retry)
assert not rules._child_session_notification_consumption_replays(raw,
    retry.model_copy(update={"model_step_id": "next-step"}))
rejected(lambda: rules._child_session_notification_consumption_record(parent=parent,
    child=child.model_copy(update={"instance_id": "replacement"}), occurrence=occurrence,
    stage=stage, consumed_at=now), SessionModelCompletionStageConflict)
consumed = rules._child_session_lifecycle_entry(parent=parent, child=child,
    records_by_type={EventType.SESSION_COMPLETED: event}, consumption_record=raw)
assert consumed.freshness.value == "consumed" and consumed.consumed_by_stage_id == "stage"
rejected(lambda: rules._child_session_notification_consumption(raw, parent=parent, child=child,
    occurrence=occurrence.model_copy(update={"source_id": "different"})), RuntimeError)
"""
    )


def test_closure_inventory_is_bounded_and_detached_without_stores():
    _without_implementations(
        """
from cayu.sessions._closure_records import (
    SESSION_CLOSURE_NATIVE_CLASSES, ClosureRecordsBuilder, ClosureRecordsTooLarge,
    validate_closure_records,
)

source = {"items": [1]}
builder = ClosureRecordsBuilder(max_records=1, max_bytes=10000)
for name in SESSION_CLOSURE_NATIVE_CLASSES:
    builder.add_class(name, [source] if name == "session" else [])
source["items"].append(2)
inventory = builder.finish()
assert inventory["records"]["session"] == [{"items": [1]}]
validated = validate_closure_records(inventory, max_records=1, max_bytes=10000)
assert validated == inventory and validated is not inventory
inventory["records"]["session"][0]["items"].append(3)
assert validated["records"]["session"] == [{"items": [1]}]
rejected(lambda: validate_closure_records(inventory, max_records=1, max_bytes=10000))
limited = ClosureRecordsBuilder(max_records=1, max_bytes=10000)
rejected(lambda: limited.add_class("session", [1, 2]), ClosureRecordsTooLarge)
rejected(limited.finish)
"""
    )


def test_closure_owner_and_progress_require_exact_replay_without_stores():
    _without_implementations(
        """
from cayu.sessions import _closure_records as rules

progress = dict(schema_version=1, root_session_id="root", plan_id="plan", policy_digest="digest",
    max_records=100, max_bytes=10000, phase="erase", descendants=[{"session_id": "child"}],
    completed=[])
receipt = dict(plan_id=rules.session_closure_target_plan_id("plan", "child"),
    root_session_id="root", target_session_id="child", operation="recursive")
rules._check_closure_lineage_owner(progress, ["child"], receipt)
rules._check_closure_lineage_owner(progress, ["unrelated"])
rejected(lambda: rules._check_closure_lineage_owner(progress, ["child"], None))
rejected(lambda: rules._check_closure_lineage_owner(progress, ["child"],
    {**receipt, "root_session_id": "other"}))
completed = {**progress, "completed": ["child"]}
rules._validate_closure_progress_update(progress, completed)
rejected(lambda: rules._validate_closure_progress_update(completed, progress))
rejected(lambda: rules._validate_closure_progress_update(progress, {**completed, "plan_id": "other"}))
tombstones = ({"child_session_id": "child", "root_session_id": "root", "plan_id": "plan",
    "original_parent_session_id": "root"},)
rules._validate_session_closure_detach_replay(tombstones, root_id="root", plan_id="plan",
    parent_session_id="root", child_session_ids=("child",))
rejected(lambda: rules._validate_session_closure_detach_replay(tombstones, root_id="root",
    plan_id="other", parent_session_id="root", child_session_ids=("child",)))
"""
    )


def test_closure_protected_effect_validation_works_without_stores():
    _without_implementations(
        """
from cayu.sessions._closure_records import require_terminal_protected_effect
from cayu.runtime._tool_effect_state import ToolEffectRecord, ToolEffectTerminal, effect_storage_key
from cayu.sessions._tool_effect_intent import ToolEffectIntent

intent = ToolEffectIntent(session_id="session", session_instance_id="instance", source_run_epoch=1,
    interaction_id="interaction", model_step_id="step", model_attempt_id="attempt",
    tool_round_id="round", tool_call_id="call", agent_name="agent", tool_name="deploy",
    idempotency_key="key", execution_profile_fingerprint="profile", schema_digest="a"*64,
    arguments_digest="b"*64)
record = ToolEffectRecord(intent=intent, revision=2, state="completed", dispatch_id="dispatch",
    terminal=ToolEffectTerminal(event_id="event", result_digest="c"*64), publication_digest="d"*64)
key = effect_storage_key(intent)
raw = record.model_dump(mode="json")
require_terminal_protected_effect("session", "instance", key, raw)
rejected(lambda: require_terminal_protected_effect("session", "other", key, raw))
rejected(lambda: require_terminal_protected_effect("session", "instance", "other", raw))
rejected(lambda: require_terminal_protected_effect("session", "instance", key,
    {**raw, "terminal": None, "state": "outcome_unknown"}))
"""
    )


def test_terminal_deletion_requires_matching_evidence_without_stores():
    _without_implementations(
        """
from tests._session_provenance import session_fixture
from cayu.events import Event, EventType
from cayu.sessions.lineage import DELETE_BLOCKED_SESSION_STATUSES, _is_durable_subagent_child
from cayu.sessions.records import SessionStatus
from cayu.sessions.terminal_evidence import _terminal_publication_delete_block_reason

session = session_fixture(id="session", agent_name="agent", provider_name="fake", model="fake",
    status=SessionStatus.COMPLETED)
assert DELETE_BLOCKED_SESSION_STATUSES == {SessionStatus.RUNNING, SessionStatus.INTERRUPTING}
assert not _is_durable_subagent_child(session)
assert _is_durable_subagent_child(session.model_copy(update={"metadata": {"subagent": {"mode": "durable"}}}))
def reason(events, checkpoint=None):
    return _terminal_publication_delete_block_reason(session=session, checkpoint=checkpoint,
        evidence_events=events)
event = Event(type=EventType.SESSION_COMPLETED, session_id=session.id)
assert reason([event]) is None
assert reason([]) is not None
assert reason([event, event]) is not None
assert reason([Event(type=EventType.SESSION_FAILED, session_id=session.id)]) is not None
assert reason([event], {"pending_session_interrupt": {}}) is not None
"""
    )


def test_effect_owner_store_annotation_resolves_without_custom_namespace():
    from typing import get_type_hints

    from cayu.runtime._tool_effect_state import ToolEffectStateOwner
    from cayu.sessions.store import SessionStore

    assert get_type_hints(ToolEffectStateOwner.__init__)["store"] is SessionStore
