"""Request authority composes without stores or execution implementations."""

from __future__ import annotations

import importlib
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import get_type_hints

import pytest

import cayu


@pytest.mark.parametrize(
    "name",
    (
        "run_request_with_runtime_invocation",
        "run_request_with_task_invocation",
        "run_request_with_runtime_session_instance_authority",
        "session_instance_id_for_run_request",
        "run_request_with_runtime_initial_transcript_authority",
        "deferred_interaction_input_for_run_request",
    ),
)
def test_request_authority_preserves_public_function_identity(name):
    canonical = getattr(importlib.import_module("cayu.sessions.requests"), name)
    assert getattr(importlib.import_module("cayu.sessions.base"), name) is canonical
    assert pickle.loads(f"ccayu.sessions.base\n{name}\n.".encode()) is canonical
    assert pickle.loads(pickle.dumps(canonical)) is canonical
    assert get_type_hints(canonical)


def _without_implementations(code: str) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys
from unittest.mock import patch

class RejectImplementations(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("cayu.storage") or fullname in {
            "cayu.sessions.base", "cayu.tasks.memory", "cayu.applications",
            "cayu.runtime._session_engine", "cayu.runtime._recovery_coordinator",
            "cayu.runtime._model_step_executor", "cayu.runtime._tool_round_executor",
        }:
            raise AssertionError(f"Request authority imported {fullname}")
sys.meta_path.insert(0, RejectImplementations())
from cayu.sessions import requests
from cayu.sessions.invocation import (
    InvocationOrigin, InvocationOriginTrust, SessionExecutionSource, TaskExecutionSource,
    TaskInvocation,
)
from cayu.tasks.creation import TaskInvocationSnapshot

def rejected(call):
    try:
        call()
    except (TypeError, ValueError):
        return
    raise AssertionError("Conflicting request authority was accepted")

instance = "12345678-1234-4234-8234-123456789abc"
source = requests.RunRequest(agent_name="agent", session_id="owned", messages=[])
"""
            + code,
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_invocation_and_task_authority_without_implementations():
    _without_implementations(
        """
origin = InvocationOrigin(trust=InvocationOriginTrust.SERVER_VERIFIED, subject="customer")
owned = requests.run_request_with_runtime_invocation(
    source, source=SessionExecutionSource.HTTP_RUN, verified_origin=origin)
assert owned is not source and source._verified_invocation_origin is None
assert owned._verified_invocation_origin == origin
assert owned._verified_invocation_origin is not origin
assert requests.copy_run_request(owned)._verified_invocation_origin == origin
rejected(lambda: requests.run_request_with_runtime_invocation(
    source, source=SessionExecutionSource.SDK_RUN, verified_origin=origin))
rejected(lambda: requests.run_request_with_runtime_invocation(source, source="http_run"))
task_source = source.model_copy(update={"task_id": "task"})
invocation = TaskInvocation(origin=origin, root_invocation_id=instance,
    source=TaskExecutionSource.HTTP_RUN)
snapshot = TaskInvocationSnapshot(id="task", session_id=None, invocation=invocation)
task_owned = requests.run_request_with_task_invocation(task_source, snapshot)
assert task_owned._runtime_invocation_source is SessionExecutionSource.TASK
assert task_owned._runtime_task_invocation == snapshot
assert task_owned._runtime_task_invocation is not snapshot
assert task_owned._runtime_task_invocation.invocation is not snapshot.invocation
assert task_source._runtime_task_invocation is None
rejected(lambda: requests.run_request_with_task_invocation(source, snapshot))
rejected(lambda: requests.run_request_with_task_invocation(
    task_source, snapshot.model_copy(update={"id": "foreign"})))
rejected(lambda: requests.run_request_with_task_invocation(
    task_source, snapshot.model_copy(update={"session_id": "foreign"})))
"""
    )


def test_session_instance_authority_without_implementations():
    _without_implementations(
        """
inspect = requests._authenticated_session_instance_id_for_run_request
resolve = requests.session_instance_id_for_run_request
with patch.object(requests, "uuid4", return_value=instance) as mint:
    assert inspect(source, session_id="owned") is None
    mint.assert_not_called()
    assert resolve(source, session_id="owned") == instance
    mint.assert_called_once_with()
    mint.reset_mock()
    owned = requests.run_request_with_runtime_session_instance_authority(
        source, session_instance_id=instance)
    assert owned is not source and source._runtime_session_instance_authority is None
    assert inspect(owned, session_id="owned") == instance
    assert resolve(requests.copy_run_request(owned), session_id="owned") == instance
    mint.assert_not_called()
    rejected(lambda: inspect(owned, session_id="foreign"))
    rejected(lambda: requests.run_request_with_runtime_session_instance_authority(
        source, session_instance_id="invalid"))
    forged = requests.copy_run_request(owned)
    forged._runtime_session_instance_authority.token = object()
    rejected(lambda: resolve(forged, session_id="owned"))
    mint.assert_not_called()
"""
    )


def test_initial_transcript_authority_without_implementations():
    _without_implementations(
        """
from cayu.messages import Message

source = requests.RunRequest(agent_name="agent", session_id="owned",
    messages=[Message.text("user", "input")])
initial = [Message.text("system", "system prompt"), *source.messages]
resolve = requests.deferred_interaction_input_for_run_request
arguments = dict(session_id="owned", interaction_id="interaction", source_messages=source.messages)
fallback = resolve(source, **arguments)
assert fallback.source_messages == source.messages
assert fallback.initial_transcript_messages is None
owned = requests.run_request_with_runtime_initial_transcript_authority(
    source, interaction_id="interaction", initial_transcript_messages=initial)
assert owned is not source and source._runtime_initial_transcript_authority is None
resolved = resolve(requests.copy_run_request(owned), **arguments)
assert resolved.initial_transcript_messages == initial
assert resolved.initial_transcript_messages[0] is not initial[0]
initial.clear()
resolved.initial_transcript_messages.clear()
assert len(resolve(owned, **arguments).initial_transcript_messages) == 2
for change in ({"session_id": "foreign"}, {"interaction_id": "foreign"},
               {"source_messages": [Message.text("user", "different")]}):
    rejected(lambda: resolve(owned, **(arguments | change)))
for forged in (object(), type("Lookalike", (), {
    "token": requests._RUNTIME_INITIAL_TRANSCRIPT_AUTHORITY_TOKEN})()):
    invalid = requests.copy_run_request(owned)
    invalid._runtime_initial_transcript_authority = forged
    rejected(lambda: resolve(invalid, **arguments))
invalid = requests.copy_run_request(owned)
invalid._runtime_initial_transcript_authority.token = object()
rejected(lambda: resolve(invalid, **arguments))
"""
    )


def test_resume_transport_authority_without_implementations():
    _without_implementations(
        """
from dataclasses import replace
from cayu.messages import Message

trace = {"traceparent": "00-" + "1" * 32 + "-" + "2" * 16 + "-01", "tracestate": "vendor=value"}
source = requests.ResumeRequest(session_id="owned", messages=[Message.text("user", "resume")], metadata={"customer": "a", **trace})
attach = requests._with_runtime_resume_transport_metadata
inspect = requests._runtime_resume_transport_metadata
assert inspect(source) == {}
owned = attach(source, trace)
assert owned is not source and owned.metadata == {"customer": "a"}
assert source.metadata == {"customer": "a", **trace}
assert inspect(owned) == inspect(requests.copy_resume_request(owned)) == trace
inspect(owned).clear()
assert inspect(owned) == trace
assert inspect(attach(owned, {})) == {}
assert attach(owned, {}).metadata == owned.metadata
for metadata in ({"customer": "a"}, {"traceparent": "different"}, {"traceparent": 1}):
    rejected(lambda: attach(source, metadata))
rejected(lambda: inspect(object()))
for forged in (object(), replace(owned._runtime_transport_metadata_authority, token=object())):
    invalid = requests.copy_resume_request(owned)
    invalid._runtime_transport_metadata_authority = forged
    assert inspect(invalid) == {}
"""
    )


def test_request_authority_private_consumers_use_canonical_owner():
    previous_owner = vars(importlib.import_module("cayu.sessions.base"))
    canonical = vars(importlib.import_module("cayu.sessions.requests"))
    for name in (
        "_authenticated_session_instance_id_for_run_request",
        "_with_runtime_resume_transport_metadata",
        "_runtime_resume_transport_metadata",
        "_RUNTIME_RESUME_TRANSPORT_METADATA_KEYS",
    ):
        assert name in canonical and name not in previous_owner
