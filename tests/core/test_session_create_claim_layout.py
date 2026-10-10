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
