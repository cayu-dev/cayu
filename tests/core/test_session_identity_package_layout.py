"""Session values compose without runtime execution or storage implementations."""

from __future__ import annotations

import importlib
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

import cayu


def _without_stores(code: str, public_module: str) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib
import importlib.abc
import pickle
import sys
from datetime import UTC, datetime, timedelta, timezone
from typing import get_type_hints

class RejectImplementations(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("cayu.storage") or fullname in {
            "cayu.sessions.base", "cayu.tasks.memory",
        } or (fullname.startswith("cayu.runtime.") and fullname not in {
            "cayu.runtime._exports", "cayu.runtime.execution_identity",
        }):
            raise AssertionError(f"Session contract imported {fullname}")
sys.meta_path.insert(0, RejectImplementations())
public = importlib.import_module(sys.argv[1])
from cayu.build_provenance import RuntimeBuildProvenance
from cayu.sessions import records
from cayu.sessions.invocation import InvocationOrigin, SessionInvocation

provenance = RuntimeBuildProvenance.unavailable("test_fixture")
invocation = SessionInvocation(
    origin=InvocationOrigin(trust="unattributed"),
    root_invocation_id="12345678-1234-4234-8234-123456789abc",
    root_session_id="session", source="sdk_run",
)
"""
            + code,
            public_module,
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "name",
    (
        "SessionIdentity",
        "SessionRuntimeIdentity",
        "copy_session_identity",
        "copy_session_runtime_identity",
    ),
)
def test_identity_historical_imports_and_pickle_globals(name):
    canonical = getattr(importlib.import_module("cayu.sessions.records"), name)
    assert getattr(importlib.import_module("cayu.sessions.base"), name) is canonical
    assert pickle.loads(f"ccayu.sessions.base\n{name}\n.".encode()) is canonical
    if name == "SessionIdentity":
        for module in ("cayu", "cayu.sessions", "cayu.runtime"):
            assert getattr(importlib.import_module(module), name) is canonical


@pytest.mark.parametrize(
    "public_module",
    (
        "cayu",
        "cayu.sessions",
        "cayu.runtime",
        "cayu.sessions.records",
    ),
)
def test_session_identity_copy_and_validation_without_implementations(public_module):
    _without_stores(
        """
identity = public.SessionIdentity(provider_name="provider", model="model",
                                  runtime_build_provenance=provenance)
runtime = records.SessionRuntimeIdentity(runtime_build_provenance=provenance)
for value, copier in (
    (identity, records.copy_session_identity),
    (runtime, records.copy_session_runtime_identity),
):
    copied = copier(value)
    assert copied == value and copied is not value
    assert copied.runtime_build_provenance is not value.runtime_build_provenance
    assert value.runtime_build_provenance is not provenance
    assert pickle.loads(pickle.dumps(value)) == value
    assert get_type_hints(type(value))
    assert type(value).model_json_schema()
    subclass = type("Subclass", (type(value),), {})
    try:
        copier(subclass.model_validate(value.model_dump(mode="json")))
    except TypeError:
        pass
    else:
        raise AssertionError("Subclass crossed the exact identity boundary")
copied = records.copy_session_identity(identity)
copied.model = "different-model"
assert identity.model == "model"
for field in ("provider_name", "model", "runtime_name", "runtime_version"):
    payload = identity.model_dump(mode="json")
    payload[field] = " "
    try:
        public.SessionIdentity.model_validate(payload)
    except ValueError:
        pass
    else:
        raise AssertionError(f"Blank {field} was accepted")
try:
    runtime.runtime_name = "changed"
except ValueError:
    pass
else:
    raise AssertionError("Runtime identity was mutable")
""",
        public_module,
    )


def test_session_instance_fingerprints_without_implementations():
    _without_stores(
        """
session = records.Session(
    id="session", instance_id="12345678-1234-4234-8234-123456789abc",
    agent_name="agent", provider_name="provider", model="model",
    causal_budget_id="budget", invocation=invocation,
    created_at=datetime(2026, 1, 1, tzinfo=UTC),
)
for fingerprint in (records._queued_dispatch_session_instance_fingerprint,
                    records._fork_source_session_instance_fingerprint):
    assert len(fingerprint(session)) == 64
    same = session.model_copy(update={"status": records.SessionStatus.COMPLETED})
    assert fingerprint(same) == fingerprint(session)
    try:
        fingerprint(session.model_dump(mode="json"))
    except TypeError:
        pass
    else:
        raise AssertionError("Unvalidated identity was accepted")
recreated = session.model_copy(update={
    "instance_id": "12345678-1234-4234-8234-123456789abd",
    "created_at": session.created_at + timedelta(seconds=1),
})
assert records._queued_dispatch_session_instance_fingerprint(recreated) != records._queued_dispatch_session_instance_fingerprint(session)
assert records._fork_source_session_instance_fingerprint(recreated) != records._fork_source_session_instance_fingerprint(session)
shifted = session.model_copy(update={
    "created_at": session.created_at.astimezone(timezone(timedelta(hours=6))),
})
assert records._queued_dispatch_session_instance_fingerprint(shifted) == records._queued_dispatch_session_instance_fingerprint(session)
""",
        "cayu.sessions.records",
    )


@pytest.mark.parametrize("name", ("SessionStateSnapshot", "SessionInvocationSnapshot"))
def test_snapshot_public_imports_and_historical_pickle_globals(name):
    canonical = getattr(importlib.import_module("cayu.sessions.records"), name)
    for module in ("cayu", "cayu.sessions", "cayu.runtime", "cayu.sessions.base"):
        assert getattr(importlib.import_module(module), name) is canonical
    assert pickle.loads(f"ccayu.sessions.base\n{name}\n.".encode()) is canonical


@pytest.mark.parametrize(
    "public_module",
    (
        "cayu",
        "cayu.sessions",
        "cayu.runtime",
        "cayu.sessions.records",
    ),
)
def test_bounded_snapshots_without_implementations(public_module):
    _without_stores(
        """
stamp = datetime(2026, 1, 1, 6, tzinfo=timezone(timedelta(hours=6)))
state = public.SessionStateSnapshot(id="session", status="pending",
                                   updated_at=stamp, last_activity_at=stamp)
assert state.updated_at == datetime(2026, 1, 1, tzinfo=UTC)
assert state.updated_at.tzinfo is UTC and state.last_activity_at.tzinfo is UTC
bound = public.SessionInvocationSnapshot(
    id="session", status="pending", invocation=invocation,
    session_instance_id="12345678-1234-4234-8234-123456789abc",
)
assert bound.invocation == invocation and bound.invocation is not invocation
for value in (state, bound):
    assert pickle.loads(pickle.dumps(value)) == value
    assert type(value).model_validate_json(value.model_dump_json()) == value
    assert get_type_hints(type(value))
    assert type(value).model_json_schema()
for value, field, bad in (
    (state, "updated_at", datetime(2026, 1, 1)),
    (state, "last_activity_at", datetime(2026, 1, 1)),
    (state, "id", " "),
    (bound, "session_instance_id", "invalid-sensitive-instance"),
    (bound, "status", "invalid"),
):
    payload = value.model_dump(mode="python")
    payload[field] = bad
    try:
        type(value).model_validate(payload)
    except ValueError as error:
        if value is bound:
            assert "invalid-sensitive-instance" not in str(error)
    else:
        raise AssertionError(f"Invalid {field} was accepted")
try:
    bound.status = records.SessionStatus.COMPLETED
except ValueError:
    pass
else:
    raise AssertionError("Invocation snapshot was mutable")
""",
        public_module,
    )
