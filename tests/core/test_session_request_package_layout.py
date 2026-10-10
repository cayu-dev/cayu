"""Request contracts and authenticated copies compose without session stores."""

from __future__ import annotations

import importlib
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

import cayu

_PUBLIC_REQUEST_NAMES = (
    "RunRequest",
    "ResumeRequest",
    "CompactSessionRequest",
    "InterruptSessionRequest",
    "ForkSessionRequest",
    "session_input_contract_evidence",
    "copy_run_request",
    "copy_resume_request",
    "copy_compact_session_request",
    "copy_interrupt_session_request",
    "copy_fork_session_request",
)


def _without_implementations(code: str, public_module: str = "cayu.sessions.requests") -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib
import importlib.abc
import pickle
import sys
from dataclasses import replace
from typing import get_type_hints

class RejectImplementations(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("cayu.storage") or fullname in {
            "cayu.sessions.base", "cayu.tasks.memory", "cayu.applications",
            "cayu.runtime._session_engine", "cayu.runtime._recovery_coordinator",
            "cayu.runtime._model_step_executor", "cayu.runtime._tool_round_executor",
        }:
            raise AssertionError(f"Request contract imported {fullname}")
sys.meta_path.insert(0, RejectImplementations())
public = importlib.import_module(sys.argv[1])
from cayu.messages import Message
from cayu.sessions import requests
from cayu.sessions.forks import ForkSourceSnapshot
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


@pytest.mark.parametrize("name", _PUBLIC_REQUEST_NAMES)
def test_request_historical_public_identity(name):
    canonical = getattr(importlib.import_module("cayu.sessions.requests"), name)
    assert getattr(importlib.import_module("cayu.sessions.base"), name) is canonical
    assert pickle.loads(f"ccayu.sessions.base\n{name}\n.".encode()) is canonical
    for module in ("cayu", "cayu.sessions", "cayu.runtime"):
        exports = importlib.import_module(module + "._exports").EXPORTS
        if name in exports:
            assert exports[name] == ("cayu.sessions.requests", name)
            assert getattr(importlib.import_module(module), name) is canonical


@pytest.mark.parametrize(
    "public_module", ("cayu", "cayu.sessions", "cayu.runtime", "cayu.sessions.requests")
)
def test_request_construction_and_copy_without_implementations(public_module):
    _without_implementations(
        """
run = public.RunRequest(agent_name="agent", messages=[Message.text("user", "input")],
                        metadata={"nested": ["retained"]})
resume = public.ResumeRequest(session_id="child", messages=[Message.text("user", "continue")])
compact = public.CompactSessionRequest(session_id="child", idempotency_key="compact",
    expected_run_epoch=1, expected_transcript_cursor=2)
interrupt = public.InterruptSessionRequest(session_id="child", reason="operator")
snapshot = ForkSourceSnapshot(source_session_id="source", source_instance_fingerprint="a" * 64,
    status="completed", run_epoch=1, transcript_cursor=2, transcript_sha256="b" * 64,
    checkpoint_sha256="c" * 64, execution_profile_fingerprint="d" * 64, causal_budget_id="budget")
fork = public.ForkSessionRequest(source_session_id="source", session_id="child",
    expected_source=snapshot, initial_invocation=resume, initial_dispatch_id="dispatch")
pairs = [(run, requests.copy_run_request), (resume, requests.copy_resume_request),
         (compact, requests.copy_compact_session_request),
         (interrupt, requests.copy_interrupt_session_request), (fork, requests.copy_fork_session_request)]
for value, copier in pairs:
    copied = copier(value)
    assert copied == value and copied is not value
    assert pickle.loads(pickle.dumps(value)) == value
    assert get_type_hints(type(value)) and type(value).model_json_schema()
    subclass = type("UntrustedRequestSubclass", (type(value),), {})
    try:
        copier(subclass.model_validate(value.model_dump(mode="python")))
    except TypeError:
        pass
    else:
        raise AssertionError("The exact request-type boundary was weakened")
for original, copied in [(run, requests.copy_run_request(run)), (resume, requests.copy_resume_request(resume))]:
    for field in ("max_steps", "limits", "thinking", "tool_completion"):
        assert (field in original.model_fields_set) == (field in copied.model_fields_set)
detached = requests.copy_run_request(run)
detached.metadata["nested"].append("changed")
assert run.metadata["nested"] == ["retained"]
assert detached.messages[0] is not run.messages[0]
assert requests.copy_fork_session_request(fork).initial_invocation is not fork.initial_invocation
marker = requests.session_input_contract_evidence(run, message_start_index=0)
assert marker.startswith("v1:0:1:original:text:sha256:")
""",
        public_module,
    )


def test_authenticated_request_copy_without_implementations():
    _without_implementations(
        """
run = requests.RunRequest(agent_name="agent", session_id="session",
                          messages=[Message.text("user", "input")])
claim = requests._RuntimeSessionCreateClaim(session_id="session", claim_id="claim")
instance = requests._RuntimeSessionInstanceAuthority(session_id="session", session_instance_id="instance")
transcript = requests._RuntimeInitialTranscriptAuthority(session_id="session", interaction_id="interaction",
    source_messages=run.messages, initial_transcript_messages=[Message.text("system", "policy"), *run.messages])
prepared = requests._RuntimePreparedSessionAuthority(
    token=requests._RUNTIME_PREPARED_SESSION_AUTHORITY_TOKEN, session_id="session", queue_task_id="task",
    dispatch_operation_id="operation", terminal_event_id="terminal", interaction_id="interaction",
    interaction_started_event_id="started", idempotency_key="key", submission_sha256="a" * 64,
    provider_name="fake", model="model", policy_evidence=None,
)
work = requests._PreparedWorkAttemptCreation("b" * 64, requests._PREPARED_WORK_ATTEMPT_CREATION_TOKEN)
capabilities = {
    "_runtime_session_create_claim": claim, "_runtime_session_instance_authority": instance,
    "_runtime_initial_transcript_authority": transcript, "_runtime_prepared_session_authority": prepared,
    "_runtime_work_attempt_creation": work,
}
for name, value in capabilities.items():
    setattr(run, name, value)
    get_type_hints(type(value))
copied = requests.copy_run_request(run)
assert all(getattr(copied, name) is value for name, value in capabilities.items())
assert all(getattr(requests.copy_run_request(pickle.loads(pickle.dumps(run))), name) is None
           for name in capabilities)
changed = run.model_copy(update={"messages": [Message.text("user", "different")]})
assert requests.copy_run_request(changed)._runtime_initial_transcript_authority is None
changed = run.model_copy(update={"session_id": "different"})
assert requests.copy_run_request(changed)._runtime_session_instance_authority is None
for name, value in capabilities.items():
    if hasattr(value, "__dataclass_fields__"):
        forged = replace(value, token=object())
    else:
        forged = pickle.loads(pickle.dumps(value))
    altered = run.model_copy()
    setattr(altered, name, forged)
    assert getattr(requests.copy_run_request(altered), name) is None
resume = requests.ResumeRequest(session_id="session", messages=[Message.text("user", "continue")])
transport = requests._RuntimeResumeTransportMetadataAuthority(
    requests._RUNTIME_RESUME_TRANSPORT_METADATA_TOKEN, (("traceparent", "transport"),))
resume._runtime_transport_metadata_authority = transport
assert requests.copy_resume_request(resume)._runtime_transport_metadata_authority is transport
resume._runtime_transport_metadata_authority = replace(transport, token=object())
assert requests.copy_resume_request(resume)._runtime_transport_metadata_authority is None
"""
    )


def test_private_request_values_have_one_owner():
    base = importlib.import_module("cayu.sessions.base")
    owner = importlib.import_module("cayu.sessions.requests")
    names = (
        "_RuntimeSessionCreateClaim",
        "_RuntimeSessionInstanceAuthority",
        "_RuntimeInitialTranscriptAuthority",
        "_RuntimePreparedSessionAuthority",
        "_RuntimeResumeTransportMetadataAuthority",
        "_PreparedWorkAttemptCreation",
        "_SessionCreateMaterial",
        "_RUNTIME_SESSION_CREATE_CLAIM_TOKEN",
        "_RUNTIME_SESSION_INSTANCE_AUTHORITY_TOKEN",
        "_RUNTIME_INITIAL_TRANSCRIPT_AUTHORITY_TOKEN",
        "_RUNTIME_PREPARED_SESSION_AUTHORITY_TOKEN",
        "_RUNTIME_RESUME_TRANSPORT_METADATA_TOKEN",
        "_PREPARED_WORK_ATTEMPT_CREATION_TOKEN",
        "_empty_run_request_authority",
        "_copy_optional_tool_capability_ceiling",
    )
    for name in names:
        assert name in vars(owner)
        assert name not in vars(base)
    assert "_copy_caller_input_message" not in vars(base)
