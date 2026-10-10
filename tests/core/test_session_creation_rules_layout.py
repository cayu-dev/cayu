"""Shared creation rules compose without storage or execution implementations."""

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
        ("records", "SessionStatusConflict"),
        ("event_delivery", "restore_persisted_event_authority"),
        ("records", "is_runtime_owned_session_metadata_key"),
        ("records", "copy_session_user_metadata"),
        ("records", "session_user_metadata"),
        ("records", "replace_session_user_metadata"),
        ("creation", "run_request_with_prepared_work_attempt_creation"),
        ("creation", "session_metadata_for_creation"),
        ("_execution_profile_rules", "SessionModelTransition"),
        ("_execution_profile_rules", "session_model_projection_cursor"),
        ("_execution_profile_rules", "execution_profile_adoption_request_fingerprint"),
        ("forks", "SessionForkSourceNotFound"),
        ("forks", "SessionForkActiveModelStageConflict"),
        ("forks", "fork_session_invocation"),
        ("forks", "effective_fork_source_execution_profile"),
        ("forks", "validate_profiled_fork_evidence"),
    ),
)
def test_creation_prerequisites_preserve_public_identity(module, name):
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
            "cayu.runtime._model_execution_selection",
        }:
            raise AssertionError(f"Shared session rules imported {fullname}")
sys.meta_path.insert(0, RejectImplementations())

def rejected(call):
    try:
        call()
    except (TypeError, ValueError):
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


def test_creation_status_contracts_without_implementations():
    _without_implementations(
        """
from cayu.sessions import records

assert issubclass(records.SessionStatusConflict, ValueError)
statuses = {records.SessionStatus.COMPLETED}
copied = records._validate_status_set(statuses, "source_statuses")
assert copied == statuses and copied is not statuses
copied.clear()
assert statuses == {records.SessionStatus.COMPLETED}
rejected(lambda: records._validate_status_set(set(), "source_statuses"))
rejected(lambda: records._validate_status_set({"completed"}, "source_statuses"))
rejected(lambda: records._validate_status_set(list(statuses), "source_statuses"))
assert records.CheckpointTransform is not None
assert records.StoreTimeCheckpointTransform is not None
"""
    )


def test_creation_event_batch_authority_without_implementations():
    _without_implementations(
        """
from cayu.events import Event, EventType, event_with_runtime_payload_authority
from cayu.sessions import event_delivery
from cayu.sessions.transcript_input import SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY

key = SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY
event = Event(type=EventType.SESSION_STARTED, session_id="session", payload={
    "parent_session_id": "parent", key: "input", "custom": {"values": [1]}})
_, untrusted = event_delivery._copy_session_event_batch("session", [event])
assert untrusted[0].payload == {"custom": {"values": [1]}}
owned = event_with_runtime_payload_authority(event, "parent_session_id", key)
_, batch = event_delivery._copy_session_event_batch("session", [owned])
assert batch[0].payload == event.payload
assert event_delivery._event_input_contract_is_runtime_owned(batch[0])
batch[0].payload["custom"]["values"].append(2)
assert owned.payload["custom"]["values"] == [1]
restored = event_delivery.restore_persisted_event_authority(
    Event.model_validate(owned.model_dump(mode="json")), input_contract_runtime_owned=True)
assert event_delivery._event_input_contract_is_runtime_owned(restored)
assert not event_delivery._event_input_contract_is_runtime_owned(
    event_delivery.restore_persisted_event_authority(event))
rejected(lambda: event_delivery._copy_session_event_batch("foreign", [owned]))
rejected(lambda: event_delivery._copy_session_event_batch("session", [owned, owned]))
"""
    )


def test_user_metadata_preserves_runtime_authority_without_implementations():
    _without_implementations(
        """
from cayu.sessions import records

current = {"customer": {"values": [1]}, "cayu:private": {"values": [2]}, "subagent": True}
replacement = records.copy_session_user_metadata({"customer": {"values": [3]}})
merged = records.replace_session_user_metadata(current, replacement)
assert merged == {"customer": {"values": [3]}, "cayu:private": {"values": [2]},
                  "subagent": True}
assert records.session_user_metadata(merged) == {"customer": {"values": [3]}}
merged["cayu:private"]["values"].clear()
merged["customer"]["values"].clear()
assert current["cayu:private"]["values"] == [2]
assert replacement["customer"]["values"] == [3]
for reserved in ("cayu:private", "subagent"):
    rejected(lambda: records.copy_session_user_metadata({reserved: "forged"}))
    rejected(lambda: records.replace_session_user_metadata(current, {reserved: "forged"}))
"""
    )


def test_creation_metadata_and_prepared_authority_without_implementations():
    _without_implementations(
        """
from datetime import UTC, datetime
from cayu.deadlines import EXECUTION_DEADLINE_METADATA_KEY, ExecutionDeadline
from cayu.sessions import creation, records, requests

identity = records.SessionIdentity(provider_name="fake", model="fake-model")
metadata = {"customer": {"values": [1]}}
deadline = ExecutionDeadline(expires_at=datetime(2099, 1, 1, tzinfo=UTC))
created = creation.session_metadata_for_creation(metadata, identity=identity,
                                               execution_deadline=deadline)
assert created[EXECUTION_DEADLINE_METADATA_KEY] == deadline.model_dump(mode="json")
assert created[records.RUNTIME_BUILD_PROVENANCE_METADATA_KEY] == (
    identity.runtime_build_provenance.model_dump(mode="json"))
created["customer"]["values"].clear()
assert metadata["customer"]["values"] == [1]
rejected(lambda: creation.session_metadata_for_creation(
    {EXECUTION_DEADLINE_METADATA_KEY: {}}, identity=identity))
request = requests.RunRequest(agent_name="agent", session_id="session", messages=[],
                              metadata=metadata, execution_deadline=deadline)
try:
    creation.run_request_with_prepared_work_attempt_creation(
        request, identity=identity, admission=object())
except RuntimeError as error:
    assert str(error) == "Prepared session creation authority returned invalid work-attempt authority."
else:
    raise AssertionError("Untyped admission authority was accepted")
request._runtime_work_attempt_creation = requests._PreparedWorkAttemptCreation(
    creation._prepared_work_attempt_creation_sha256(request, identity), object())
rejected(lambda: creation.session_metadata_for_creation(
    metadata, identity=identity, execution_deadline=deadline, prepared_request=request))
assert metadata == {"customer": {"values": [1]}}
"""
    )


def test_profile_admission_and_projection_without_implementations():
    _without_implementations(
        """
from cayu.build_provenance import (
    RuntimeBuildArtifactKind, RuntimeBuildProvenance, RuntimeBuildProvenanceOrigin,
)
from cayu.runtime.execution_profiles import build_execution_profile_identity
from cayu.sessions import _execution_profile_rules as rules, records
from cayu.sessions._execution_profile_checkpoint import execution_profile_session_metadata
from cayu.sessions._model_failover import ModelTarget
from cayu.sessions.invocation import InvocationOrigin, SessionInvocation

def profile(model):
    return build_execution_profile_identity(runtime_name="cayu", runtime_version="test",
        provider_name="fake", model=model, durable_system_prompt="instructions", direct_tools=(),
        tool_catalogue_revision="sha256:" + "a" * 64,
        runtime_build_provenance=RuntimeBuildProvenance.from_artifact_digest(
            origin=RuntimeBuildProvenanceOrigin.EXPLICIT_MANIFEST,
            artifact_kind=RuntimeBuildArtifactKind.OTHER, artifact_digest="b" * 64))

expected, candidate = profile("first"), profile("second")
session = records.Session(id="session", agent_name="agent", provider_name="fake", model="first",
    runtime_version="test", causal_budget_id="session", metadata={
        "cayu:execution_profile": execution_profile_session_metadata(expected),
        records.RUNTIME_BUILD_PROVENANCE_METADATA_KEY: expected.runtime_build_provenance.model_dump(mode="json")},
    invocation=SessionInvocation(origin=InvocationOrigin(trust="unattributed"),
        root_invocation_id="12345678-1234-4234-8234-123456789abc", root_session_id="session", source="sdk_run"))
assert rules._validate_execution_profile_admission(session, candidate_profile=expected,
                                                  model_transition=None, decision=None) is None
rejected(lambda: rules._validate_execution_profile_admission(session, candidate_profile=candidate,
                                                            model_transition=None, decision=None))
copied = rules._copy_optional_execution_profile(expected)
assert copied == expected and copied is not expected
assert rules.session_model_projection_cursor(session) == 0
metadata = rules._session_metadata_with_model_projection(session.metadata,
    target=ModelTarget(provider_name="fake", model="first"), transcript_cursor=7)
assert rules.session_model_projection_cursor(session.model_copy(update={"metadata": metadata})) == 7
assert "cayu:model_target_projection" not in session.metadata
rejected(lambda: rules._session_metadata_with_model_projection(session.metadata,
    target=ModelTarget(provider_name="fake", model="first"), transcript_cursor=True))
"""
    )


def test_profile_adoption_fingerprint_preserves_semantic_authority():
    _without_implementations(
        """
from cayu.approvals.actors import ResolutionActor, ResolutionActorSource
from cayu.execution_profiles import ExecutionProfileAdoptionIntent
from cayu.messages import Message
from cayu.sessions import _execution_profile_rules as rules, requests
from cayu.vaults.redaction import SecretRedactor

intent = ExecutionProfileAdoptionIntent(idempotency_key="adopt", reason="change profile",
    requested_by=ResolutionActor(subject="operator", source=ResolutionActorSource.REQUEST))
messages = [Message.text("user", "continue")]
request = requests.ResumeRequest(session_id="session", messages=messages, profile_adoption=intent)
redactor = SecretRedactor()
fingerprint = rules.execution_profile_adoption_request_fingerprint(request, redactor=redactor)
assert len(fingerprint) == 64
assert fingerprint == rules.execution_profile_adoption_request_fingerprint(
    requests.copy_resume_request(request), redactor=redactor)
changed = request.model_copy(update={"metadata": {"customer": "different"}})
assert fingerprint != rules.execution_profile_adoption_request_fingerprint(changed, redactor=redactor)
rejected(lambda: rules.execution_profile_adoption_request_fingerprint(
    requests.ResumeRequest(session_id="session", messages=messages), redactor=redactor))
rejected(lambda: rules.execution_profile_adoption_request_fingerprint(request, redactor=object()))
"""
    )


def test_fork_source_validation_without_implementations():
    _without_implementations(
        """
from cayu.sessions import forks, records
from cayu.sessions.invocation import InvocationOrigin, SessionInvocation

source = records.Session(id="source", agent_name="agent", provider_name="fake", model="model",
    causal_budget_id="source", status=records.SessionStatus.COMPLETED,
    invocation=SessionInvocation(origin=InvocationOrigin(trust="unattributed"),
        root_invocation_id="12345678-1234-4234-8234-123456789abc", root_session_id="source", source="sdk_run"))
child = source.model_copy(update={"id": "child", "parent_session_id": "source",
                                  "invocation": forks.fork_session_invocation(source)})
source_id, detached, statuses, cursor = forks._prepare_session_fork_request(
    source_session_id="source", fork=child, source_statuses={source.status}, transcript_cursor=0)
assert source_id == source.id and detached == child and detached is not child
assert cursor == 0
assert forks._validate_session_fork_source(source_session=source, source_session_id=source_id,
    fork=detached, allowed_statuses=statuses, expected_source_run_epoch=source.run_epoch) is source
rejected(lambda: forks._validate_session_fork_source(source_session=source,
    source_session_id=source_id, fork=detached, allowed_statuses=statuses,
    expected_source_run_epoch=source.run_epoch + 1))
rejected(lambda: forks._prepare_session_fork_request(source_session_id="foreign", fork=child,
    source_statuses=statuses, transcript_cursor=cursor))
try:
    forks._validate_session_fork_source(source_session=None, source_session_id=source_id,
        fork=detached, allowed_statuses=statuses, expected_source_run_epoch=0)
except forks.SessionForkSourceNotFound:
    pass
else:
    raise AssertionError("Missing fork source was accepted")
"""
    )


def test_fork_profile_and_model_selection_without_execution():
    _without_implementations(
        """
from cayu.build_provenance import RuntimeBuildProvenance
from cayu.runtime.execution_profiles import build_execution_profile_identity
from cayu.sessions import forks, records
from cayu.sessions._execution_profile_checkpoint import (
    execution_profile_session_metadata, model_failover_progress_for_session,
)
from cayu.sessions.invocation import InvocationOrigin, SessionInvocation

profile = build_execution_profile_identity(runtime_name="cayu", runtime_version="test",
    provider_name="fake", model="model", durable_system_prompt="instructions", direct_tools=(),
    tool_catalogue_revision="sha256:" + "a" * 64,
    runtime_build_provenance=RuntimeBuildProvenance.unavailable("test_fixture"))
source = records.Session(id="source", agent_name="agent", provider_name="fake", model="model",
    runtime_version="test", causal_budget_id="source", metadata={
        "cayu:execution_profile": execution_profile_session_metadata(profile),
        records.RUNTIME_BUILD_PROVENANCE_METADATA_KEY: profile.runtime_build_provenance.model_dump(mode="json")},
    invocation=SessionInvocation(origin=InvocationOrigin(trust="unattributed"),
        root_invocation_id="12345678-1234-4234-8234-123456789abc", root_session_id="source", source="sdk_run"))
kind, selected, active = forks.effective_fork_source_execution_profile(source, None)
assert kind is forks.ForkExecutionProfileSource.SESSION_EXPECTED
assert selected == profile and active is None
for checkpoint in (None, {"custom": {"value": 1}}):
    assert model_failover_progress_for_session(session=source, execution_profile=profile,
                                              checkpoint=checkpoint) is None
"""
    )
