"""Fork evidence composes without loading session stores or runtime execution."""

from __future__ import annotations

import importlib
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

import cayu
from cayu.build_provenance import RuntimeBuildProvenance
from cayu.runtime.execution_profiles import build_execution_profile_identity


def _without_stores(code: str, public_module: str) -> None:
    profile = build_execution_profile_identity(
        runtime_name="cayu",
        runtime_version="test",
        provider_name="fake",
        model="fake-model",
        durable_system_prompt="fork contract fixture",
        direct_tools=(),
        tool_catalogue_revision="sha256:" + "c" * 64,
        runtime_build_provenance=RuntimeBuildProvenance.unavailable("test_fixture"),
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib
import importlib.abc
import json
import pickle
import sys
from typing import get_type_hints

class RejectImplementations(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("cayu.storage") or fullname in {
            "cayu.sessions.base", "cayu.tasks.memory",
        } or (fullname.startswith("cayu.runtime.") and fullname not in {
            "cayu.runtime._exports", "cayu.runtime.execution_identity",
        }):
            raise AssertionError(f"Fork contract imported {fullname}")
sys.meta_path.insert(0, RejectImplementations())
public = importlib.import_module(sys.argv[1])
from cayu.execution_profiles import ExecutionProfileIdentity
from cayu.sessions import forks
from cayu.sessions._execution_profile_checkpoint import (
    EXECUTION_PROFILE_METADATA_KEY, execution_profile_session_metadata,
)
from cayu.sessions.invocation import InvocationOrigin, SessionInvocation
from cayu.sessions.records import RUNTIME_BUILD_PROVENANCE_METADATA_KEY, Session

profile = ExecutionProfileIdentity.model_validate(json.load(sys.stdin))
invocation = SessionInvocation(
    origin=InvocationOrigin(trust="unattributed"),
    root_invocation_id="12345678-1234-4234-8234-123456789abc",
    root_session_id="child", source="sdk_run",
)
session = Session(
    id="child", parent_session_id="source", agent_name="agent",
    provider_name="fake", model="fake-model", environment_name="sandbox",
    causal_budget_id="budget", invocation=invocation,
    metadata={
        EXECUTION_PROFILE_METADATA_KEY: execution_profile_session_metadata(profile),
        RUNTIME_BUILD_PROVENANCE_METADATA_KEY: profile.runtime_build_provenance.model_dump(mode="json"),
    },
)
"""
            + code,
            public_module,
        ],
        input=profile.model_dump_json(),
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "name",
    (
        "ForkSystemPromptPolicy",
        "ForkExecutionProfileSelection",
        "ForkExecutionProfileSource",
        "ForkSourceSnapshot",
        "fork_source_state_sha256",
        "ForkExecutionProfileDecisionRecord",
        "SessionForkEnvironmentAllocationOwner",
        "SessionForkProfileRelationship",
        "ProfiledSessionForkResult",
        "copy_profiled_session_fork_result",
        "session_fork_profile_relationship",
    ),
)
def test_fork_historical_imports_and_pickle_globals(name):
    canonical = getattr(importlib.import_module("cayu.sessions.forks"), name)
    assert getattr(importlib.import_module("cayu.sessions.base"), name) is canonical
    assert pickle.loads(f"ccayu.sessions.base\n{name}\n.".encode()) is canonical
    for module in ("cayu", "cayu.sessions", "cayu.runtime"):
        exports = importlib.import_module(module + "._exports").EXPORTS
        if name in exports:
            assert exports[name] == ("cayu.sessions.forks", name)
            assert getattr(importlib.import_module(module), name) is canonical


@pytest.mark.parametrize(
    "public_module", ("cayu", "cayu.sessions", "cayu.runtime", "cayu.sessions.forks")
)
def test_fork_evidence_without_implementations(public_module):
    _without_stores(
        """
snapshot = public.ForkSourceSnapshot(
    source_session_id="source", source_instance_fingerprint="a" * 64,
    status="completed", run_epoch=2, transcript_cursor=3,
    transcript_sha256="b" * 64, checkpoint_sha256="c" * 64,
    execution_profile_fingerprint=profile.fingerprint, causal_budget_id="budget",
)
fingerprint = forks.fork_source_state_sha256(snapshot)
assert len(fingerprint) == 64
assert forks.fork_source_state_sha256(snapshot.model_copy(update={
    "source_session_id": "redacted", "causal_budget_id": "redacted",
})) == fingerprint
relationship = forks.SessionForkProfileRelationship(
    schema_version=2, request_sha256="d" * 64, source_state_sha256=fingerprint,
    source_session_id="source", child_session_id="child", child_agent_name="agent",
    child_provider_name="fake", child_model="fake-model", child_environment_name="sandbox",
    source_status="completed", source_run_epoch=2, source_profile_source="session_expected",
    source_profile=profile, selected_profile=profile, copy_checkpoint=True,
    system_prompt_policy="inherit_source", selection="inherit_parent",
    source_environment_allocation_owners=(), fork_event_id="fork-event",
)
session.metadata[forks.FORK_EXECUTION_PROFILE_METADATA_KEY] = relationship.model_dump(mode="json")
assert forks.session_fork_profile_relationship(session) == relationship
session.metadata[forks.FORK_EXECUTION_PROFILE_METADATA_KEY]["child_session_id"] = "wrong-child"
try:
    forks.session_fork_profile_relationship(session)
except ValueError as exc:
    assert "conflicts with lineage" in str(exc)
else:
    raise AssertionError("Unrelated fork evidence was accepted")
for value in (snapshot, relationship):
    assert pickle.loads(pickle.dumps(value)) == value
    assert get_type_hints(type(value))
    assert type(value).model_json_schema()
""",
        public_module,
    )


def test_fork_acknowledgement_copy_without_implementations():
    _without_stores(
        """
from cayu.events import Event, EventType

events = [Event(type=EventType.SESSION_FORKED, session_id="child", payload={"nested": []}),
          Event(type=EventType.TARGETED_TOOL_GRANT_FORK_RESET, session_id="child")]
result = forks.ProfiledSessionForkResult(session=session, events=events)
copied = forks.copy_profiled_session_fork_result(result)
assert copied == result and copied is not result
assert copied.session is not result.session
assert copied.events[0] is not result.events[0]
copied.events[0].payload["nested"].append("changed")
assert result.events[0].payload["nested"] == events[0].payload["nested"] == []
malformed = result.model_copy()
object.__getattribute__(malformed, "__dict__")["unexpected"] = True
try:
    forks.copy_profiled_session_fork_result(malformed)
except TypeError:
    pass
else:
    raise AssertionError("Malformed acknowledgement state was accepted")
""",
        "cayu.sessions.forks",
    )
