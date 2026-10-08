"""Session records and evidence compose without loading native stores."""

from __future__ import annotations

import importlib
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

import cayu

_PUBLIC_RECORDS = (
    "Session",
    "SessionStatus",
    "EventRecord",
    "TranscriptRecord",
    "RunnerObservedEventIdentity",
    "RUNTIME_BUILD_PROVENANCE_METADATA_KEY",
)


def test_session_record_imports_preserve_public_identity_and_pickle_globals():
    from cayu.sessions import base, records

    surfaces = [importlib.import_module(name) for name in ("cayu", "cayu.sessions", "cayu.runtime")]
    for name in _PUBLIC_RECORDS:
        canonical = getattr(records, name)
        assert getattr(base, name) is canonical
        assert all(getattr(surface, name) is canonical for surface in surfaces)
        if isinstance(canonical, type):
            assert pickle.loads(f"ccayu.sessions.base\n{name}\n.".encode()) is canonical
    assert base.copy_session is records.copy_session
    assert base.runtime_build_provenance_from_session_metadata is (
        records.runtime_build_provenance_from_session_metadata
    )


@pytest.mark.parametrize("public_module", ("cayu", "cayu.sessions", "cayu.runtime"))
def test_public_session_records_work_without_loading_concrete_stores(public_module):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib
import importlib.abc
import pickle
import sys
from typing import get_type_hints

blocked = {
    "cayu.sessions.base", "cayu.storage.sqlite", "cayu.storage.postgres",
    "cayu.storage.tasks_sqlite", "cayu.tasks.memory",
}
class RejectStores(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked:
            raise AssertionError(f"Session records imported {fullname}")
sys.meta_path.insert(0, RejectStores())

public = importlib.import_module(sys.argv[1])
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.sessions.invocation import InvocationOrigin, SessionInvocation
from cayu.sessions.records import copy_session

session = public.Session(
    id="session", agent_name="agent", provider_name="provider", model="model",
    invocation=SessionInvocation(
        origin=InvocationOrigin(trust="unattributed"),
        root_invocation_id="12345678-1234-4234-8234-123456789abc",
        root_session_id="session", source="sdk_run",
    ),
    metadata={"nested": {"value": "original"}},
)
copied = copy_session(session)
copied.metadata["nested"]["value"] = "changed"
assert session.metadata["nested"]["value"] == "original"
assert copied.instance_id == session.instance_id
assert session.causal_budget_id == session.id
assert session.runtime_build_provenance.fingerprint is None
event = public.EventRecord(sequence=1, event=Event(
    type=EventType.SESSION_STARTED, session_id=session.id,
))
transcript = public.TranscriptRecord(index=0, message=Message.text("user", "hello"))
identity = public.RunnerObservedEventIdentity(session.id, 1, EventType.SESSION_STARTED)
for value in (session, event, transcript, identity):
    restored = pickle.loads(pickle.dumps(value))
    assert restored == value and type(restored) is type(value)
    assert get_type_hints(type(value))
for cls in (public.Session, public.EventRecord, public.TranscriptRecord):
    assert cls.model_json_schema()
assert not blocked.intersection(sys.modules)
""",
            public_module,
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
