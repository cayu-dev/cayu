"""Creation-claim contracts compose independently of session implementations."""

from __future__ import annotations

import importlib
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

import cayu

_CONTRACT_NAMES = (
    "RuntimeSessionCreateClaimReferenceKey",
    "RuntimeSessionCreateClaimReference",
    "RuntimeSessionCreateClaimAuthenticationDisposition",
    "RuntimeSessionCreateClaimAuthentication",
)


def _without_implementations(code: str, public_module: str) -> None:
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

class RejectImplementations(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("cayu.storage") or fullname in {
            "cayu.sessions.base", "cayu.tasks.memory", "cayu.applications",
            "cayu.runtime._session_engine", "cayu.runtime._recovery_coordinator",
            "cayu.runtime._model_step_executor", "cayu.runtime._tool_round_executor",
        }:
            raise AssertionError(f"Creation claim imported {fullname}")
sys.meta_path.insert(0, RejectImplementations())
public = importlib.import_module(sys.argv[1])
from cayu.sessions import creation_claims as claims
from cayu.sessions.records import SessionStatus
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


@pytest.mark.parametrize("name", _CONTRACT_NAMES)
def test_creation_claim_contract_historical_public_identity(name):
    canonical = getattr(importlib.import_module("cayu.sessions.creation_claims"), name)
    assert getattr(importlib.import_module("cayu.sessions.base"), name) is canonical
    assert pickle.loads(f"ccayu.sessions.base\n{name}\n.".encode()) is canonical
    for module in ("cayu", "cayu.sessions", "cayu.runtime"):
        exports = importlib.import_module(module + "._exports").EXPORTS
        if name in exports:
            assert exports[name] == ("cayu.sessions.creation_claims", name)
            assert getattr(importlib.import_module(module), name) is canonical


def test_creation_claim_values_without_implementations():
    _without_implementations(
        """
key = public.RuntimeSessionCreateClaimReferenceKey(key_id="test", secret=b"private-canary" * 3)
assert "private-canary" not in repr(key)
reference = public.RuntimeSessionCreateClaimReference(
    session_id="session", operation_id="operation", request_authority_key_id=key.key_id,
    request_authority_hmac_sha256="a" * 64, claim_id="b" * 64,
)
disposition = public.RuntimeSessionCreateClaimAuthenticationDisposition
missing = public.RuntimeSessionCreateClaimAuthentication(disposition=disposition.MISSING_SESSION)
matching = public.RuntimeSessionCreateClaimAuthentication(
    disposition=disposition.MATCHING_SESSION, session_status=SessionStatus.RUNNING,
    transient_input_authenticated=True,
)
assert not missing.matches and matching.matches
for value in (key, reference, missing, matching, disposition.IDENTITY_CONFLICT):
    assert pickle.loads(pickle.dumps(value)) == value
for value in (key, reference, missing):
    assert get_type_hints(type(value))
assert type(reference).model_validate_json(reference.model_dump_json()) == reference
assert type(matching).model_validate_json(matching.model_dump_json()) == matching
assert type(reference).model_json_schema() and type(missing).model_json_schema()
for change in ({"schema_version": True}, {"claim_id": "A" * 64},
               {"request_authority_hmac_sha256": "short"}, {"operation_id": "x" * 257}):
    try:
        type(reference).model_validate(reference.model_dump() | change)
    except ValueError:
        pass
    else:
        raise AssertionError("Malformed claim reference was accepted")
for values in ({"disposition": disposition.MISSING_SESSION, "session_status": "running"},
               {"disposition": disposition.FOREIGN_SESSION, "session_status": "running",
                "transient_input_authenticated": True}):
    try:
        type(missing).model_validate(values)
    except ValueError:
        pass
    else:
        raise AssertionError("Conflicting authentication evidence was accepted")
""",
        "cayu.sessions.creation_claims",
    )


def test_raw_digest_validation_has_one_independent_owner():
    _without_implementations(
        """
from cayu.sessions.authority import _require_raw_sha256_digest
assert _require_raw_sha256_digest("0123456789abcdef" * 4) is None
for value in (None, True, 64, b"a" * 64, "A" * 64, "a" * 63, "a" * 65):
    try:
        _require_raw_sha256_digest(value)
    except ValueError as error:
        assert error.args == ()
    else:
        raise AssertionError("Malformed raw digest was accepted")
""",
        "cayu.sessions.creation_claims",
    )
    assert "_require_raw_sha256_digest" not in vars(importlib.import_module("cayu.sessions.base"))


def test_creation_claim_authentication_without_implementations():
    _without_implementations(
        """
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.sessions.invocation import InvocationOrigin, SessionExecutionSource, SessionInvocation
from cayu.sessions.records import RUNTIME_BUILD_PROVENANCE_METADATA_KEY, Session, SessionIdentity
from cayu.sessions.requests import RunRequest, run_request_with_runtime_generated_authority
from cayu.sessions.transcript_input import DeferredInteractionInput

key = claims.RuntimeSessionCreateClaimReferenceKey(key_id="test", secret=b"k" * 32)
source = run_request_with_runtime_generated_authority(
    RunRequest(agent_name="agent", session_id="owned", messages=[Message.text("user", "input")]),
    "session_id",
)
reference = claims.runtime_session_create_claim_reference(source, operation_id="operation", key=key)
reference = type(reference).model_validate_json(reference.model_dump_json())
attached, claim = claims.run_request_with_runtime_session_create_claim_reference(
    source, reference, operation_id="operation", key=key,
)
prepared = claims.apply_runtime_session_create_claim(attached)
identity = SessionIdentity(provider_name="fake", model="model")
started = Event(type=EventType.INTERACTION_STARTED, session_id="owned", interaction_id="interaction")
claims.bind_runtime_session_create_claim(prepared, identity=identity, interaction_started_event=started)
session = Session(
    id="owned", agent_name="agent", provider_name="fake", model="model", status="running",
    invocation=SessionInvocation(origin=InvocationOrigin(trust="unattributed"),
        root_invocation_id="12345678-1234-4234-8234-123456789abc", root_session_id="owned",
        source=SessionExecutionSource.SDK_RUN),
    metadata=prepared.metadata | {RUNTIME_BUILD_PROVENANCE_METADATA_KEY:
                                 identity.runtime_build_provenance.model_dump(mode="json")},
)
deferred = DeferredInteractionInput(interaction_id="interaction", source_messages=source.messages)
disposition = claims.RuntimeSessionCreateClaimAuthenticationDisposition
def authenticate(value, pending=deferred, secret_key=key):
    return claims.authenticate_runtime_session_create_claim_reference(
        value, pending, claim, reference, request=attached, operation_id="operation",
        parent_session=None, key=secret_key,
    )
assert authenticate(None).disposition is disposition.MISSING_SESSION
matched = authenticate(session)
assert matched.matches and matched.transient_input_authenticated
assert claims.session_has_runtime_create_claim(session, claim)
assert claims.session_matches_runtime_create_claim(
    session, deferred, claim, request=attached, parent_session=None)
assert claims.session_matches_reconstructed_runtime_create_claim(
    session, deferred, claim, request=attached, parent_session=None)
assert claims.session_invocation_matches_run_request(session, request=attached, parent_session=None)
assert claims.strip_runtime_session_create_claim_before_redaction(prepared).metadata == source.metadata
assert authenticate(session, None).disposition is disposition.INCOMPLETE_EVIDENCE
terminal = session.model_copy(update={"status": SessionStatus.COMPLETED})
assert authenticate(terminal, None).matches
assert not authenticate(terminal, None).transient_input_authenticated
tampered = deferred.model_copy(update={"source_messages": [Message.text("user", "different")]})
assert authenticate(session, tampered).disposition is disposition.TAMPERED_EVIDENCE
assert authenticate(session.model_copy(update={"agent_name": "foreign"})).disposition is disposition.FOREIGN_SESSION
wrong_key = claims.RuntimeSessionCreateClaimReferenceKey(key_id="test", secret=b"x" * 32)
assert authenticate(session, secret_key=wrong_key).disposition is disposition.IDENTITY_CONFLICT
for name in ("runtime_session_create_claim_reference", "bind_runtime_session_create_claim",
             "authenticate_runtime_session_create_claim_reference"):
    assert get_type_hints(getattr(claims, name))
""",
        "cayu.sessions.creation_claims",
    )


def test_request_attestation_and_prepared_authority_without_implementations():
    _without_implementations(
        """
from dataclasses import replace
from cayu.sessions import requests

source = requests.RunRequest(agent_name="agent", session_id="owned", messages=[])
assert not requests.run_request_authority_is_runtime_generated(source, field_name="session_id", value="owned")
owned = requests.run_request_with_runtime_generated_authority(source, "session_id")
assert requests.run_request_authority_is_runtime_generated(owned, field_name="session_id", value="owned")
assert not requests.run_request_authority_is_runtime_generated(owned, field_name="session_id", value="other")
prepared = requests.run_request_with_prepared_session_authority(
    owned, session_id="owned", queue_task_id="task", dispatch_operation_id="operation",
    terminal_event_id="terminal", interaction_id="interaction", interaction_started_event_id="started",
    idempotency_key="key", submission_sha256="a" * 64, provider_name="fake", model="model",
    policy_evidence=None,
)
authority = requests.runtime_prepared_session_authority(prepared)
assert authority is not None
copied = requests.copy_run_request(prepared)
assert requests.runtime_prepared_session_authority(copied) is authority
fingerprint = requests._run_request_invocation_lifecycle_authority_sha256
assert fingerprint(prepared) == fingerprint(copied) != fingerprint(owned)
copied._runtime_prepared_session_authority = replace(authority, token=object())
assert requests.runtime_prepared_session_authority(copied) is None
assert copied._runtime_prepared_session_authority is None
copied = prepared.model_copy(update={"session_id": "other"})
assert requests.runtime_prepared_session_authority(copied) is None
assert requests.runtime_prepared_session_authority(pickle.loads(pickle.dumps(prepared))) is None
""",
        "cayu.sessions.requests",
    )


@pytest.mark.parametrize(
    "name,owner",
    (
        ("_runtime_session_create_reference_operation_id", "creation_claims"),
        ("_runtime_session_create_reference_request_hmac_sha256", "creation_claims"),
        ("_runtime_session_create_reference_claim_id", "creation_claims"),
        ("_session_create_claim_record", "creation_claims"),
        ("_runtime_session_create_authentication", "creation_claims"),
        ("_run_request_invocation_lifecycle_authority_sha256", "requests"),
    ),
)
def test_private_creation_rules_have_one_owner(name, owner):
    assert name in vars(importlib.import_module("cayu.sessions." + owner))
    assert name not in vars(importlib.import_module("cayu.sessions.base"))
