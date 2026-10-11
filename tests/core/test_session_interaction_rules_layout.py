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
        ("_interaction_transitions", "QueuedInteractionProfileHandoff"),
        ("_interaction_transitions", "queued_interaction_profile_handoff_evidence"),
        ("_interaction_transitions", "InteractionTransitionResult"),
        ("_interaction_transitions", "InteractionTransitionSpec"),
        ("_interaction_transitions", "InteractionTransitionReceiptResult"),
        ("_interaction_transitions", "SessionInvocationAdmission"),
        ("_interaction_transitions", "copy_interaction_transition_spec"),
        ("_invocation_scope", "resolve_interaction_attribution"),
        ("_invocation_scope", "attribute_event_to_current_interaction"),
        ("_invocation_scope", "attribute_events_to_current_interaction"),
        ("creation", "session_invocation_for_run_request"),
    ),
)
def test_interaction_prerequisites_preserve_public_identity(module, name):
    canonical = getattr(importlib.import_module(f"cayu.sessions.{module}"), name)
    assert getattr(importlib.import_module("cayu.sessions.base"), name) is canonical
    assert pickle.loads(f"ccayu.sessions.base\n{name}\n.".encode()) is canonical
    assert pickle.loads(pickle.dumps(canonical)) is canonical


@pytest.mark.parametrize(
    "name", ("SessionAccessSelector", "SessionAccessRule", "SessionAccessScope")
)
def test_access_rules_preserve_public_identity(name):
    canonical = getattr(importlib.import_module("cayu.sessions._access_rules"), name)
    assert getattr(importlib.import_module("cayu.sessions.access"), name) is canonical
    assert pickle.loads(f"ccayu.sessions.access\n{name}\n.".encode()) is canonical
    assert pickle.loads(pickle.dumps(canonical)) is canonical


def test_access_denial_preserves_public_identity():
    from cayu._resource_access_errors import ResourceAccessDenied
    from cayu.sessions.access import SessionAccessDenied

    assert SessionAccessDenied is ResourceAccessDenied


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
            "cayu.sessions.base", "cayu.sessions.store", "cayu.sessions.memory", "cayu.tasks.memory", "cayu.applications",
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


_SESSION_FIXTURE = """
from cayu.build_provenance import RuntimeBuildProvenance
from cayu.events import Event, EventType
from cayu.runtime.execution_profiles import build_execution_profile_identity
from cayu.sessions._execution_profile_checkpoint import (
    ActiveInvocationExecutionProfile, execution_profile_session_metadata,
)
from cayu.sessions.invocation import InvocationOrigin, SessionInvocation
from cayu.sessions.records import Session, SessionStatus
from cayu.tools.exposure import ToolCapabilityCeiling, session_metadata_with_tool_capability_ceiling

profile = build_execution_profile_identity(runtime_name="cayu", runtime_version="test",
    provider_name="fake", model="model", durable_system_prompt="instructions", direct_tools=(),
    tool_catalogue_revision="sha256:" + "a" * 64,
    runtime_build_provenance=RuntimeBuildProvenance.unavailable("test_fixture"))
session = Session(id="session", agent_name="agent", provider_name="fake", model="model",
    runtime_version="test", causal_budget_id="session", status=SessionStatus.COMPLETED,
    run_epoch=1, metadata={"cayu:execution_profile": execution_profile_session_metadata(profile),
        "cayu:runtime_build_provenance": profile.runtime_build_provenance.model_dump(mode="json")},
    invocation=SessionInvocation(origin=InvocationOrigin(trust="unattributed"),
        root_invocation_id="12345678-1234-4234-8234-123456789abc", root_session_id="session", source="sdk_run"))
session.metadata = session_metadata_with_tool_capability_ceiling(
    session.metadata, ToolCapabilityCeiling(tool_names=()))
active = ActiveInvocationExecutionProfile(session_id="session", interaction_id="first",
    run_epoch=1, profile=profile)
"""


def test_interaction_summaries_do_not_load_invocation_contracts():
    _without_implementations(
        """
class RejectInvocationContracts(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {
            "cayu.sessions._interaction_transitions",
            "cayu.sessions._execution_profile_rules",
            "cayu.sessions.requests",
        }:
            raise AssertionError(f"Interaction summaries imported {fullname}")
sys.meta_path.insert(0, RejectInvocationContracts())
from cayu.sessions.interactions import InteractionStatus, InteractionSummaryEvidence
assert InteractionStatus.COMPLETED.value == "completed"
assert InteractionSummaryEvidence.__module__ == "cayu.sessions.interactions"
"""
    )


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


def test_interaction_transition_contracts_without_implementations():
    _without_implementations(
        _SESSION_FIXTURE
        + """
from cayu.sessions import _interaction_transitions as rules

event = Event(type=EventType.INTERACTION_COMPLETED, session_id=session.id,
              interaction_id="first", payload={"nested": {"value": 1}})
spec = rules.InteractionTransitionSpec(event=event, from_statuses=(SessionStatus.RUNNING,),
    to_status=SessionStatus.COMPLETED)
copied = rules.copy_interaction_transition_spec(spec)
event.payload["nested"]["value"] = 2
assert copied == spec and copied is not spec
assert spec.event.payload == {"nested": {"value": 1}}
receipt = rules.InteractionTransitionReceiptResult(session=session, transition=spec,
    status_changed=True)
assert receipt.replayed and receipt.session is not session
rejected(lambda: rules.InteractionTransitionReceiptResult(session=session,
    transition=spec, status_changed=False))
rejected(lambda: rules.InteractionTransitionSpec(event=event,
    from_statuses=(SessionStatus.RUNNING,), to_status=SessionStatus.RUNNING))
rejected(lambda: rules.InteractionTransitionSpec(event=event,
    from_statuses=(SessionStatus.RUNNING, SessionStatus.RUNNING),
    to_status=SessionStatus.COMPLETED))
"""
    )


def test_invocation_admission_and_queued_handoff_without_implementations():
    _without_implementations(
        _SESSION_FIXTURE
        + """
from cayu.messages import Message
from cayu.sessions import _interaction_transitions as rules
from cayu.tools.exposure import ToolCapabilityCeiling

started = Event(type=EventType.INTERACTION_STARTED, session_id=session.id, interaction_id="next")
fields = dict(from_statuses=frozenset({SessionStatus.COMPLETED}),
    checkpoint_transform=lambda session, checkpoint: checkpoint, execution_profile=profile,
    interaction_source_messages=(Message.text("user", "continue"),),
    tool_capability_ceiling=ToolCapabilityCeiling(tool_names=()))
admission = rules.SessionInvocationAdmission(**fields, interaction_started_event=started)
assert admission.execution_profile == profile and admission.execution_profile is not profile
assert admission.interaction_started_event is not started
rejected(lambda: rules.SessionInvocationAdmission(**fields, interaction_started_event=started,
    continued_interaction_id="first"))
rejected(lambda: rules.SessionInvocationAdmission(**fields, continued_interaction_id="first"))
continued = rules.SessionInvocationAdmission(**fields, continued_interaction_id="first",
    expected_active_invocation_profile=active, defer_interaction_source=True)
assert continued.continued_interaction_id == "first"
target = active.model_copy(update={"interaction_id": "next"})
handoff = rules.QueuedInteractionProfileHandoff(expected_session_instance_id=session.instance_id,
    predecessor_settlement_event_id="settled", expected_active_profile=active,
    target_active_profile=target)
started.payload[rules.QUEUED_INTERACTION_PROFILE_HANDOFF_PAYLOAD_KEY] = (
    rules.queued_interaction_profile_handoff_evidence(handoff))
copied = rules._copy_queued_interaction_profile_handoff(session.id, "next", "next", started, handoff)
assert copied == handoff and copied is not handoff
rejected(lambda: rules._copy_queued_interaction_profile_handoff(
    session.id, "wrong-delivery", "next", started, handoff))
"""
    )


def test_run_fence_retirement_crosses_copied_contexts_without_implementations():
    _without_implementations(
        _SESSION_FIXTURE
        + """
import asyncio
from cayu.sessions import _invocation_scope as scope
from cayu.sessions.authority import SessionRunFenced

async def scenario():
    owner = scope._activate_owned_session_run_fence(session)
    retired = []
    owner.on_retire(lambda: retired.append("retired"))
    proceed = asyncio.Event()
    ready = asyncio.Event()
    async def child():
        assert scope._current_session_run_epoch(session.id) == 1
        ready.set()
        await proceed.wait()
        assert scope._current_session_run_epoch(session.id) is None
    task = asyncio.create_task(child())
    try:
        await ready.wait()
        rejected(lambda: scope._assert_session_run_epoch_value(session.id, 2), SessionRunFenced)
        assert owner.retire() and not owner.retire()
        assert retired == ["retired"]
    finally:
        proceed.set()
        await task
        scope._deactivate_session_run_fence(session.id)
    rejected(lambda: owner.activate().__enter__(), RuntimeError)
    assert scope._current_recovery_cleanup_identity() is None
asyncio.run(scenario())
"""
    )


def test_interaction_context_preserves_history_and_restores_outer_scope():
    _without_implementations(
        _SESSION_FIXTURE
        + """
import asyncio
from cayu.sessions import _invocation_scope as scope
from cayu.sessions._interaction_transitions import InteractionTransitionSpec

async def scenario():
    scope._activate_owned_session_run_fence(session)
    scope._activate_session_interaction(session.id, "first")
    captured = scope._SessionRunFenceContext.current_or_new()
    scope._deactivate_session_interaction(session.id)
    scope._deactivate_session_run_fence(session.id)
    with captured.activate():
        assert scope._SessionRunFenceContext.current_or_new() is captured
        assert scope._current_recovery_cleanup_identity() == (session.id, session.instance_id, 1)
        with captured.activate():
            assert scope.resolve_interaction_attribution(session.id, scope.INHERIT_INTERACTION) == "first"
            assert scope.resolve_interaction_attribution(session.id, None) is None
        event = Event(type=EventType.MODEL_STARTED, session_id=session.id)
        assert scope.attribute_event_to_current_interaction(event).interaction_id == "first"
        assert event.interaction_id is None
        terminal = Event(type=EventType.SESSION_COMPLETED, session_id=session.id)
        assert scope.attribute_event_to_current_interaction(terminal).interaction_id is None
        spec = InteractionTransitionSpec(event=Event(type=EventType.INTERACTION_COMPLETED,
            session_id=session.id, interaction_id="first"),
            from_statuses=(SessionStatus.RUNNING,), to_status=SessionStatus.COMPLETED)
        scope._mark_session_interaction_settled(session.id, "first", spec)
        scope._mark_session_invocation_terminal_event(terminal)
        scope._close_session_interaction(session.id)
        assert scope._current_session_interaction_id(session.id) is None
        assert scope._latest_session_invocation_interaction_is_settled(session.id)
        retained = scope._current_session_invocation_settlement_transition(session.id)
        assert retained == spec and retained is not spec
        assert scope._current_session_invocation_terminal_event(session.id) == terminal
    assert scope._current_session_run_epoch(session.id) is None
    assert scope._current_session_invocation_interaction_ids(session.id) == ()
    with captured.activate():
        assert scope._latest_session_invocation_interaction_is_settled(session.id)
        scope._activate_session_interaction(session.id, "next")
        assert not scope._latest_session_invocation_interaction_is_settled(session.id)
        assert scope._current_session_invocation_settlement_transition(session.id) is None
        scope._deactivate_session_interaction(session.id)
        assert scope._current_session_invocation_terminal_event(session.id) is None
        scope._deactivate_session_run_fence(session.id)
asyncio.run(scenario())
"""
    )


def test_transition_receipts_replay_after_exact_release_without_implementations():
    _without_implementations(
        _SESSION_FIXTURE
        + """
from cayu.sessions import _interaction_receipts as rules, _invocation_lifecycle as lifecycle
from cayu.sessions.authority import SessionRunFenced
from cayu.sessions.checkpoints import INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY
from cayu.sessions._interaction_transitions import InteractionTransitionSpec

event = Event(type=EventType.INTERACTION_COMPLETED, session_id=session.id, interaction_id="first")
spec = InteractionTransitionSpec(event=event, from_statuses=(SessionStatus.RUNNING,),
    to_status=SessionStatus.COMPLETED)
record = rules._interaction_transition_receipt_record(session=session, event=event,
    from_statuses={SessionStatus.RUNNING}, to_status=SessionStatus.COMPLETED,
    only_if_no_queued_messages=False, model_completion_stage_settlement=None,
    status_changed=True, invocation_session_instance_id=session.instance_id,
    invocation_active_profile=active)
receipt = rules._reconstruct_interaction_transition_receipt(record, transition=spec)
rules._require_invocation_release_settlement_record(record, current_session=session,
    transition=spec, expected_session_instance_id=session.instance_id,
    expected_active_invocation_profile=active)
advanced = session.model_copy(update={"run_epoch": 2})
authority = dict(expected_session_instance_id=session.instance_id,
                 expected_active_invocation_profile=active)
rejected(lambda: rules._validate_interaction_transition_receipt_authority(receipt,
    current_session=advanced, current_checkpoint=None, **authority), SessionRunFenced)
command = lifecycle.ReleaseInvocationCommand(session_id=session.id,
    expected_session_instance_id=session.instance_id, expected_run_epoch=1,
    expected_active_profile=active, settlement_transition=spec)
release = lifecycle._invocation_lifecycle_command_receipt(command,
    active_profile=active, result_session=advanced)
ledger = lifecycle._InvocationLifecycleReceiptLedger(receipts=(release,))
checkpoint = {INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY: ledger.model_dump(mode="json")}
rules._validate_interaction_transition_receipt_authority(receipt,
    current_session=advanced, current_checkpoint=checkpoint, **authority)
rejected(lambda: rules._validate_interaction_transition_receipt_authority(receipt,
    current_session=advanced.model_copy(update={"instance_id": "replacement"}),
    current_checkpoint=checkpoint, **authority), SessionRunFenced)
record["event"]["payload"]["forged"] = True
rejected(lambda: rules._load_interaction_transition_receipt(record), RuntimeError)
"""
    )


def test_terminal_receipts_require_runtime_event_authority_without_implementations():
    _without_implementations(
        _SESSION_FIXTURE
        + """
from cayu.events import event_with_runtime_envelope_authority
from cayu.sessions import _interaction_receipts as rules
from cayu.sessions.authority import SessionRunFenced
from cayu.sessions.checkpoints import ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY

checkpoint = {ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY: active.model_dump(mode="json")}
event = Event(type=EventType.SESSION_COMPLETED, session_id=session.id)
assert rules._invocation_terminal_event_receipt_record(
    session=session, checkpoint=checkpoint, event=event) is None
owned = event_with_runtime_envelope_authority(event, "session_id")
key, record = rules._invocation_terminal_event_receipt_record(
    session=session, checkpoint=checkpoint, event=owned)
assert key == rules._invocation_terminal_event_storage_key(event.id)
rules._require_invocation_release_terminal_session_event(record, owned,
    current_session=session, expected_event=owned, expected_session_instance_id=session.instance_id,
    expected_active_invocation_profile=active)
rejected(lambda: rules._require_invocation_release_terminal_session_event(record, None,
    current_session=session, expected_event=owned, expected_session_instance_id=session.instance_id,
    expected_active_invocation_profile=active), SessionRunFenced)
record["event"]["payload"]["forged"] = True
rejected(lambda: rules._load_invocation_terminal_event_receipt(record), RuntimeError)
"""
    )


def test_paired_terminal_decision_requires_exact_profile_without_implementations():
    _without_implementations(
        _SESSION_FIXTURE
        + """
from datetime import UTC, datetime
from cayu.sessions import _interaction_receipts as rules
from cayu.sessions._invocation_terminal_decision import (
    InvocationTerminalOutcome, build_invocation_terminal_decision,
    checkpoint_with_invocation_terminal_decision, settled_invocation_terminal_decision_from_checkpoint,
)
from cayu.sessions.authority import SessionRunFenced
from cayu.sessions.checkpoints import ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY

recovery_id = "model-recovery:" + "b" * 64
decision = build_invocation_terminal_decision(outcome=InvocationTerminalOutcome.FAILED,
    session_id=session.id, session_instance_id=session.instance_id, run_epoch=1,
    profile_interaction_id="first", interaction_id="first",
    execution_profile_fingerprint=profile.fingerprint,
    interaction_event_id=recovery_id + ":interaction_failed",
    terminal_event_id=recovery_id + ":session_failed", observed_at=datetime(2026, 1, 1, tzinfo=UTC),
    terminal_payload={}, model_recovery_id=recovery_id)
checkpoint = checkpoint_with_invocation_terminal_decision(
    {ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY: active.model_dump(mode="json")}, decision)
rejected(lambda: rules._checkpoint_after_exact_invocation_terminal_decision(
    checkpoint, session=session, expected=None), SessionRunFenced)
rejected(lambda: rules._checkpoint_after_exact_invocation_terminal_decision(
    checkpoint, session=session.model_copy(update={"run_epoch": 2}), expected=decision), SessionRunFenced)
settled = rules._checkpoint_after_exact_invocation_terminal_decision(
    checkpoint, session=session, expected=decision)
assert settled_invocation_terminal_decision_from_checkpoint(settled) == decision
assert rules._checkpoint_after_exact_invocation_terminal_decision(None, session=session,
                                                                expected=None) is None
"""
    )


def test_scoped_creation_provenance_without_implementations():
    _without_implementations(
        """
from cayu.sessions import _access_rules as rules, creation
from cayu.sessions.requests import RunRequest

wide = rules.SessionAccessRule(selectors=(rules.SessionAccessSelector(key="tenant", values=("a", "b")),))
narrow = rules.SessionAccessRule(selectors=(rules.SessionAccessSelector(key="tenant", values=("a",)),))
admitted = rules.SessionAccessScope(read=(wide,), create=(wide,))
current = rules.SessionAccessScope(read=(narrow,), create=(narrow,))
bounds = rules._SessionAccessBounds(admitted, current)
token = rules._creation_bounds.set(bounds)
try:
    request = RunRequest(agent_name="agent", session_id="session", messages=[], labels={"tenant": "a"})
    invocation = creation.session_invocation_for_run_request(request,
        session_id="session", parent_session=None)
    assert invocation.root_session_id == "session"
    assert invocation.source.value == "sdk_run"
    foreign = request.model_copy(update={"labels": {"tenant": "b"}})
    rejected(lambda: creation.session_invocation_for_run_request(foreign,
        session_id="session", parent_session=None), rules.SessionAccessDenied)
    assert admitted.matches("read", {"tenant": "b"})
    assert not bounds.matches({"tenant": "b"})
finally:
    rules._creation_bounds.reset(token)
assert rules._creation_bounds.get() is None
"""
    )
