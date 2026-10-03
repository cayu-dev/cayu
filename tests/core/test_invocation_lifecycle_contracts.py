"""Invocation contracts retain identity independently of live command dispatch."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

import cayu
from cayu.build_provenance import (
    RuntimeBuildArtifactKind,
    RuntimeBuildProvenance,
    RuntimeBuildProvenanceOrigin,
)
from cayu.runtime import _invocation_lifecycle as runtime
from cayu.runtime.execution_profiles import build_execution_profile_identity
from cayu.sessions import _invocation_lifecycle as contracts
from cayu.sessions._execution_profile_checkpoint import ActiveInvocationExecutionProfile
from cayu.sessions.authority import SessionRunFenced
from cayu.sessions.base import InMemorySessionStore


def _active_profile():
    return ActiveInvocationExecutionProfile(
        session_id="session",
        interaction_id="interaction",
        run_epoch=1,
        profile=build_execution_profile_identity(
            runtime_name="cayu",
            runtime_version="test",
            provider_name="fake",
            model="fake-model",
            durable_system_prompt="invocation contract fixture",
            direct_tools=(),
            tool_catalogue_revision="sha256:" + "c" * 64,
            runtime_build_provenance=RuntimeBuildProvenance.from_artifact_digest(
                origin=RuntimeBuildProvenanceOrigin.EXPLICIT_MANIFEST,
                artifact_kind=RuntimeBuildArtifactKind.OTHER,
                artifact_digest="a" * 64,
            ),
        ),
    )


@pytest.mark.parametrize(
    "module_name",
    ("_invocation_lifecycle", "authority", "_durable_operation_ownership", "invocation_release"),
)
def test_legacy_invocation_contract_imports_and_pickles_share_the_canonical_owner(module_name):
    canonical = importlib.import_module(f"cayu.sessions.{module_name}")
    legacy = importlib.import_module(f"cayu.runtime.{module_name}")
    for name, value in vars(canonical).items():
        if getattr(value, "__module__", None) != canonical.__name__:
            continue
        assert getattr(legacy, name) is value
        if inspect.isclass(value) or inspect.isfunction(value):
            assert pickle.loads(f"c{legacy.__name__}\n{name}\n.".encode()) is value
    for name in (
        "INVOCATION_LIFECYCLE_COMMAND_VERSION",
        "INVOCATION_LIFECYCLE_RECEIPT_LEDGER_MAX_ITEMS",
        "INVOCATION_LIFECYCLE_RECEIPT_LEDGER_MAX_BYTES",
        "_RELEASE_CLEANUP_AUTHORITY_TOKEN",
        "_RELEASE_STORE_AUTHORITY_TOKEN",
    ):
        assert getattr(runtime, name) is getattr(contracts, name)
    assert cayu.SessionRunFenced is SessionRunFenced
    assert cayu.runtime.CreateInvocationCommand is contracts.CreateInvocationCommand
    assert cayu.runtime.InvocationContext is runtime.InvocationContext


def test_runtime_release_authority_survives_validated_copy_but_not_tampering_or_pickle():
    command = contracts.ReleaseInvocationCommand(
        session_id="session",
        expected_session_instance_id="00000000-0000-4000-8000-000000000001",
        expected_run_epoch=1,
        expected_active_profile=_active_profile(),
        recovery_claim_id="recovery-claim",
    )
    with pytest.raises(SessionRunFenced, match="authenticated cleanup"):
        contracts.require_invocation_release_store_authority(command)
    cleaned = runtime._release_invocation_command_with_cleanup_authority(command)
    prepared = asyncio.run(
        runtime._prepare_release_invocation_command_for_store(InMemorySessionStore(), cleaned)
    )
    copied = contracts.copy_invocation_lifecycle_command(prepared)
    assert copied is not prepared
    contracts.require_invocation_release_store_authority(copied)
    runtime.require_invocation_release_store_authority(
        runtime.copy_invocation_lifecycle_command(copied)
    )
    for untrusted in (
        copied.model_copy(update={"recovery_claim_id": "different-claim"}),
        pickle.loads(pickle.dumps(copied)),
    ):
        detached = contracts.copy_invocation_lifecycle_command(untrusted)
        with pytest.raises(SessionRunFenced, match="authenticated cleanup"):
            contracts.require_invocation_release_store_authority(detached)


@pytest.mark.parametrize("temporary_service", (False, True))
def test_invocation_contracts_validate_without_loading_live_execution_owners(temporary_service):
    script = """
import importlib.abc
import json
import sys

blocked = {
    "cayu.runtime._invocation_lifecycle", "cayu.runtime._checkpoint_store",
    "cayu.runtime._session_engine", "cayu.runtime._recovery_coordinator",
    "cayu.runtime._runtime_records", "cayu.storage.sqlite", "cayu.storage.postgres",
    "cayu.runtime._session_continuation", "cayu.runtime._temporary_continuation",
}
class RejectLiveExecution(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked:
            raise AssertionError(f"Invocation contracts imported {fullname}")
sys.meta_path.insert(0, RejectLiveExecution())
from cayu.runtime import AdmitInvocationCommand
from cayu.sessions import _invocation_lifecycle as contracts
from cayu.sessions._execution_profile_checkpoint import ActiveInvocationExecutionProfile
from cayu.sessions._session_continuation import (
    continuation_admission_digest, continuation_admission_inputs,
)
from cayu.events import Event, EventType
from cayu.tools.exposure import ToolCapabilityCeiling

active = ActiveInvocationExecutionProfile.model_validate(json.load(sys.stdin))
command = AdmitInvocationCommand(
    session_id=active.session_id,
    expected_session_instance_id="00000000-0000-4000-8000-000000000001",
    expected_statuses=("pending",), expected_run_epoch=0,
    expected_checkpoint_sha256="a" * 64, target_active_profile=active,
    tool_capability_ceiling=ToolCapabilityCeiling(tool_names=()),
    interaction_started_event=Event(type=EventType.INTERACTION_STARTED,
        session_id=active.session_id, interaction_id=active.interaction_id,
        agent_name="assistant"),
    **({"temporary_service_operation_key": "session-continuation:service:" + "b" * 64,
        "participant_permit_operation": "operation",
        "participant_permit_commitment": "c" * 64} if sys.argv[1] == "True" else {}),
)
assert type(command) is contracts.AdmitInvocationCommand
assert contracts.copy_invocation_lifecycle_command(command) == command
assert continuation_admission_digest(command) == (
    contracts.invocation_admission_command_sha256(command))
assert continuation_admission_inputs(command)[1] == active.profile.fingerprint
assert contracts.invocation_admission_command_sha256(command) == (
    contracts.invocation_admission_command_sha256(
        contracts.copy_invocation_lifecycle_command(command)))
assert contracts.invocation_lifecycle_receipt_history_present(None) is False
assert not blocked.intersection(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(temporary_service)],
        input=_active_profile().model_dump_json(),
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_durable_authority_values_work_without_runtime_or_store_implementations():
    script = """
import importlib.abc
import sys
class RejectRuntimeAndStores(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname == "cayu.runtime" or fullname.startswith("cayu.runtime.")
                or fullname == "cayu.storage" or fullname.startswith("cayu.storage.")
                or fullname == "cayu.sessions.base"):
            raise AssertionError(f"Durable authority imported {fullname}")
sys.meta_path.insert(0, RejectRuntimeAndStores())
from cayu import SessionRunFenced
from cayu.sessions.authority import checkpoint_value_authority
from cayu.sessions._durable_operation_ownership import DurableOperationOwnershipState
from cayu.sessions.invocation_release import InvocationReleaseEvidence
assert SessionRunFenced.__module__ == "cayu.sessions.authority"
assert checkpoint_value_authority({"a": 1, "b": 2}, "value") == (
    checkpoint_value_authority({"b": 2, "a": 1}, "value"))
assert DurableOperationOwnershipState.RELEASED == "released"
evidence = InvocationReleaseEvidence(session_id="session", session_instance_id="instance",
    interaction_id="interaction", command_identity="release:session:instance:1",
    command_sha256="a"*64, record_sha256="b"*64, profile_fingerprint="c"*64,
    run_epoch=1, released_run_epoch=2)
assert evidence.released_run_epoch == evidence.run_epoch + 1
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
