"""Profile values and checkpoint rules compose independently of runtime."""

from __future__ import annotations

import gc
import os
import pickle
import subprocess
import sys
import weakref
from pathlib import Path

import pytest

import cayu
from cayu import execution_profiles as contracts
from cayu.approvals import actors
from cayu.approvals import tools as approval_tools
from cayu.build_provenance import (
    RuntimeBuildArtifactKind,
    RuntimeBuildProvenance,
    RuntimeBuildProvenanceOrigin,
)
from cayu.events import Event, EventType
from cayu.runtime import execution_profiles as legacy
from cayu.sessions import _execution_profile_checkpoint as checkpoint

_CONTRACT_CLASSES = (
    "ExecutionProfileAdoptionIntent",
    "ExecutionProfileAuthorityDecision",
    "ExecutionProfileComponentClass",
    "ExecutionProfileComponentIdentity",
    "ExecutionProfileDecision",
    "ExecutionProfileDecisionKind",
    "ExecutionProfileIdentity",
    "ExecutionProfileIdentityAvailability",
    "ExecutionProfileIdentityStrength",
    "ExecutionProfilePolicyAction",
    "ExecutionProfilePolicyRequest",
    "ExecutionProfilePolicyResult",
    "ExecutionProfileRejectionResult",
    "ModelFailoverCandidateProfile",
    "ModelFailoverProfileBinding",
)


def _profile():
    return legacy.build_execution_profile_identity(
        runtime_name="cayu",
        runtime_version="test",
        provider_name="fake",
        model="fake-model",
        durable_system_prompt="profile contract fixture",
        direct_tools=(),
        tool_catalogue_revision="sha256:" + "c" * 64,
        runtime_build_provenance=RuntimeBuildProvenance.from_artifact_digest(
            origin=RuntimeBuildProvenanceOrigin.EXPLICIT_MANIFEST,
            artifact_kind=RuntimeBuildArtifactKind.OTHER,
            artifact_digest="a" * 64,
        ),
    )


def test_profile_consumers_share_the_canonical_contracts_and_checkpoint_rules():
    for name in _CONTRACT_CLASSES:
        assert getattr(legacy, name) is getattr(contracts, name)
    for name in (
        "ExecutionProfileIdentity",
        "ExecutionProfileDecision",
        "ExecutionProfilePolicyRequest",
        "ExecutionProfilePolicyResult",
    ):
        assert getattr(cayu, name) is getattr(contracts, name)
        assert getattr(cayu.runtime, name) is getattr(contracts, name)
    for name in (
        "ActiveInvocationExecutionProfile",
        "EXECUTION_PROFILE_METADATA_KEY",
        "active_invocation_execution_profile_from_checkpoint",
        "active_invocation_execution_profile_is_released",
        "active_invocation_execution_profile_matches_session_epoch",
        "checkpoint_with_active_invocation_execution_profile",
        "execution_profile_from_session_metadata",
        "execution_profile_baseline_from_session_metadata",
        "execution_profile_metadata_after_adoption",
        "execution_profile_session_metadata",
    ):
        assert getattr(legacy, name) is getattr(checkpoint, name)
    assert (
        cayu.runtime.ActiveInvocationExecutionProfile is checkpoint.ActiveInvocationExecutionProfile
    )
    assert (
        legacy.ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY
        == checkpoint.ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY
    )
    assert type(_profile()) is contracts.ExecutionProfileIdentity


@pytest.mark.parametrize("name", (*_CONTRACT_CLASSES, "ActiveInvocationExecutionProfile"))
def test_legacy_pickled_profile_classes_resolve_to_the_canonical_owner(name):
    owner = checkpoint if name == "ActiveInvocationExecutionProfile" else contracts
    model = getattr(owner, name)
    encoded = pickle.dumps(model, protocol=0)
    historical = encoded.replace(owner.__name__.encode(), legacy.__name__.encode())
    assert historical != encoded
    assert pickle.loads(historical) is model


@pytest.mark.parametrize("name", ("ResolutionActor", "ResolutionActorSource"))
def test_legacy_approval_actor_imports_and_pickles_keep_the_shared_identity(name):
    model = getattr(actors, name)
    assert getattr(approval_tools, name) is model
    assert getattr(cayu, name) is model
    encoded = pickle.dumps(model, protocol=0)
    historical = encoded.replace(actors.__name__.encode(), approval_tools.__name__.encode())
    assert historical != encoded
    assert pickle.loads(historical) is model


def test_runtime_decision_attestation_uses_canonical_values_without_copying_authority():
    profile = _profile()
    values = {
        "kind": contracts.ExecutionProfileDecisionKind.EXACT_REUSE,
        "expected_profile": profile,
        "candidate_profile": profile,
        "changed_component_classes": (),
        "policy_identity": "test:profile:v1",
        "policy_reason": "Exact identity",
        "authority_decision": contracts.ExecutionProfileAuthorityDecision.NOT_REQUIRED,
        "idempotency_identity": "profile-decision",
        "actor": None,
        "reason": "Exact identity",
    }
    decision = contracts.ExecutionProfileDecision(
        **values,
        event=Event(
            type=EventType.SESSION_EXECUTION_PROFILE_DECIDED,
            session_id="session",
            agent_name="assistant",
            payload=contracts.execution_profile_decision_payload(**values),
        ),
    )
    assert not legacy._has_runtime_execution_profile_decision_authority(decision)
    assert legacy._with_runtime_execution_profile_decision_authority(decision) is decision
    assert legacy._has_runtime_execution_profile_decision_authority(decision)
    copied = contracts.copy_execution_profile_decision(decision)
    assert copied == decision
    assert not legacy._has_runtime_execution_profile_decision_authority(copied)
    reference = weakref.ref(decision)
    identity = id(decision)
    del decision
    gc.collect()
    assert reference() is None
    assert identity not in legacy._RUNTIME_EXECUTION_PROFILE_DECISIONS


def test_profile_contracts_and_checkpoint_rules_work_without_runtime_or_stores():
    script = """
import importlib.abc
import json
import sys

class RejectRuntimeAndStores(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname == "cayu.runtime" or fullname.startswith("cayu.runtime.")
                or fullname == "cayu.storage" or fullname.startswith("cayu.storage.")
                or fullname == "cayu.sessions.base"):
            raise AssertionError(f"Profile contracts imported {fullname}")

sys.meta_path.insert(0, RejectRuntimeAndStores())
from cayu import ExecutionProfileIdentity, ResolutionActor, ResolutionActorSource
from cayu import execution_profiles as contracts
from cayu.approvals import actors
from cayu.sessions import _execution_profile_checkpoint as checkpoint

assert ExecutionProfileIdentity is contracts.ExecutionProfileIdentity
assert ResolutionActor is actors.ResolutionActor
actor = ResolutionActor(subject="operator", source=ResolutionActorSource.REQUEST,
                        claims={"roles": ["operator"]})
intent = contracts.ExecutionProfileAdoptionIntent(
    idempotency_key="adoption", reason="New tool", requested_by=actor,
)
actor.claims["roles"].append("changed")
assert intent.requested_by.claims == {"roles": ["operator"]}
assert actors.resolution_actor_payload(intent.requested_by) == {
    "subject": "operator", "tenant": None, "source": "request",
}
profile = ExecutionProfileIdentity.model_validate(json.load(sys.stdin))
metadata = {checkpoint.EXECUTION_PROFILE_METADATA_KEY:
            checkpoint.execution_profile_session_metadata(profile)}
assert checkpoint.execution_profile_from_session_metadata(metadata) == profile
assert checkpoint.execution_profile_baseline_from_session_metadata(metadata) == profile
candidate = contracts.execution_profile_with_tool_capability_ceiling(profile, ("tool",))
adopted = checkpoint.execution_profile_metadata_after_adoption(metadata, candidate)
assert checkpoint.execution_profile_from_session_metadata(adopted) == candidate
assert checkpoint.execution_profile_baseline_from_session_metadata(adopted) == profile
assert checkpoint.execution_profile_from_session_metadata(metadata) == profile
assert contracts.changed_execution_profile_components(profile, candidate) == (
    contracts.ExecutionProfileComponentClass.TOOL_VIEW_GRANTS,
)
saved = checkpoint.checkpoint_with_active_invocation_execution_profile(
    None, session_id="session", interaction_id="interaction", run_epoch=1, profile=profile,
)
active = checkpoint.active_invocation_execution_profile_from_checkpoint(saved)
assert type(active) is checkpoint.ActiveInvocationExecutionProfile and active.profile == profile
assert checkpoint.active_invocation_execution_profile_matches_session_epoch(
    active, session_id="session", run_epoch=2,
)
assert checkpoint.active_invocation_execution_profile_is_released(
    active, session_id="session", run_epoch=2,
)
rebound = checkpoint.checkpoint_with_active_invocation_execution_profile(
    saved, session_id="session", interaction_id="interaction", run_epoch=2,
    profile=profile, expected=active,
)
assert checkpoint.active_invocation_execution_profile_from_checkpoint(saved) == active
assert checkpoint.active_invocation_execution_profile_from_checkpoint(rebound).run_epoch == 2
assert not any(name == "cayu.runtime" or name.startswith("cayu.runtime.")
               or name == "cayu.storage" or name.startswith("cayu.storage.")
               or name == "cayu.sessions.base" for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        input=_profile().model_dump_json(),
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
