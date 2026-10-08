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


def test_run_operation_marker_parses_without_runtime_or_stores():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            r"""
import importlib.abc
import sys

class RejectRuntimeAndStores(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("cayu.runtime", "cayu.storage")) or fullname == "cayu.sessions.base":
            raise AssertionError(f"Run-operation evidence imported {fullname}")
sys.meta_path.insert(0, RejectRuntimeAndStores())

from cayu.sessions._terminal_evidence import _session_run_operation_from_checkpoint

assert _session_run_operation_from_checkpoint(None) is None
assert _session_run_operation_from_checkpoint({}) is None
marker = {
    "version": 1, "operation_id": "operation", "run_epoch": 2,
    "terminal_event_id": "terminal", "queue_task_id": "task",
}
checkpoint = {"session_run_operation": marker}
parsed = _session_run_operation_from_checkpoint(checkpoint)
assert (parsed.operation_id, parsed.run_epoch, parsed.terminal_event_id, parsed.queue_task_id) == (
    "operation", 2, "terminal", "task",
)
marker["operation_id"] = "changed"
assert parsed.operation_id == "operation"
for field, value in (
    ("version", 2), ("run_epoch", True), ("run_epoch", 0),
    ("operation_id", " "), ("operation_id", "bad\ud800"),
    ("terminal_event_id", None),
):
    malformed = {**marker, field: value}
    try:
        _session_run_operation_from_checkpoint({"session_run_operation": malformed})
    except ValueError:
        pass
    else:
        raise AssertionError(f"Accepted malformed marker field: {field}")
""",
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_terminal_evidence_imports_preserve_public_identity_and_pickle_globals():
    from cayu.sessions import base, terminal_evidence

    surfaces = [importlib.import_module(name) for name in ("cayu", "cayu.sessions", "cayu.runtime")]
    names = (
        "TerminalPublicationMarker",
        "TerminalSessionEvidence",
        "TerminalSessionEvidenceBoundary",
        "TerminalSessionEvidenceError",
        "TerminalSessionEvidenceErrorCode",
        "TerminalSessionEvidenceLimits",
        "TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_EVENTS",
        "TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_TRANSCRIPT_RECORDS",
        "TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_RECORD_BYTES",
        "TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_TOTAL_BYTES",
        "TERMINAL_SESSION_EVIDENCE_HARD_MAX_EVENTS",
        "TERMINAL_SESSION_EVIDENCE_HARD_MAX_TRANSCRIPT_RECORDS",
        "TERMINAL_SESSION_EVIDENCE_HARD_MAX_RECORD_BYTES",
        "TERMINAL_SESSION_EVIDENCE_HARD_MAX_TOTAL_BYTES",
    )
    for name in names:
        canonical = getattr(terminal_evidence, name)
        assert getattr(base, name) is canonical
        assert all(getattr(surface, name) is canonical for surface in surfaces)
        if isinstance(canonical, type):
            assert pickle.loads(f"ccayu.sessions.base\n{name}\n.".encode()) is canonical
    assert base.copy_terminal_session_evidence is terminal_evidence.copy_terminal_session_evidence


@pytest.mark.parametrize("public_module", ("cayu", "cayu.sessions", "cayu.runtime"))
def test_terminal_evidence_composes_without_loading_concrete_stores(public_module):
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

class RejectStores(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("cayu.storage") or fullname in {
            "cayu.sessions.base", "cayu.tasks.memory",
        }:
            raise AssertionError(f"Terminal evidence imported {fullname}")
sys.meta_path.insert(0, RejectStores())

public = importlib.import_module(sys.argv[1])
from cayu._validation import compact_json_utf8_size
from cayu.events import Event, EventType
from cayu.sessions.invocation import InvocationOrigin, SessionInvocation
from cayu.sessions.terminal_evidence import (
    _assemble_terminal_session_evidence, copy_terminal_session_evidence,
)

session = public.Session(
    id="session", agent_name="agent", provider_name="provider", model="model",
    invocation=SessionInvocation(
        origin=InvocationOrigin(trust="unattributed"),
        root_invocation_id="12345678-1234-4234-8234-123456789abc",
        root_session_id="session", source="sdk_run",
    ),
    status=public.SessionStatus.COMPLETED,
    metadata={"nested": {"value": "original"}},
)
terminal = public.EventRecord(sequence=1, event=Event(
    type=EventType.SESSION_COMPLETED, session_id=session.id,
))
kwargs = dict(session=session, marker=None, terminal_record=terminal,
              events=(terminal,), transcript=())
evidence = _assemble_terminal_session_evidence(
    **kwargs, limits=public.TerminalSessionEvidenceLimits(),
)
assert type(evidence) is public.TerminalSessionEvidence
assert evidence.boundary.total_bytes == compact_json_utf8_size(evidence.model_dump(mode="json"))
assert evidence.terminal_event == terminal
copied = copy_terminal_session_evidence(evidence)
copied.session.metadata["nested"]["value"] = "changed"
assert evidence.session.metadata["nested"]["value"] == "original"
restored = pickle.loads(pickle.dumps(evidence))
assert restored == evidence and type(restored) is type(evidence)
assert get_type_hints(public.TerminalSessionEvidence)
assert public.TerminalSessionEvidence.model_json_schema()
try:
    _assemble_terminal_session_evidence(
        **kwargs, limits=public.TerminalSessionEvidenceLimits(max_record_bytes=1),
    )
except public.TerminalSessionEvidenceError as exc:
    assert exc.code is public.TerminalSessionEvidenceErrorCode.RECORD_BYTES_EXCEEDED
    assert exc.limit == 1
else:
    raise AssertionError("Terminal evidence ignored its record-byte limit")
""",
            public_module,
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
