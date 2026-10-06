"""Saved cleanup evidence can be read without loading execution owners."""

from __future__ import annotations

import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

import cayu
from cayu._validation import canonical_durable_json_bytes
from cayu.sessions import _completion_finalization as reader


def _marker():
    return {
        "version": 1,
        "outcome": "completed",
        "environment_name": "coding",
        "binding_generation_id": "binding-1",
        "execution_profile_fingerprint": "profile-1",
        "binding_state": {"files": [{"path": "result.txt"}]},
    }


def test_completion_reader_imports_without_runtime_environment_or_store_owners():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
blocked = ("cayu.runtime", "cayu.environments", "cayu.storage", "cayu.sessions.base")
class BlockOwners:
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            raise AssertionError(fullname)
sys.meta_path.insert(0, BlockOwners())
from cayu.sessions import _completion_finalization as reader
marker = {"version": 1, "outcome": "failed", "environment_name": "coding",
    "binding_generation_id": "binding-1", "execution_profile_fingerprint": "profile-1",
    "binding_state": {"files": ["result.txt"]}}
checkpoint = {reader.PENDING_COMPLETION_FINALIZATION_CHECKPOINT_KEY: marker}
result = reader.pending_completion_finalization_from_checkpoint(checkpoint)
assert result == marker and result is not marker
result["binding_state"]["files"].clear()
assert marker["binding_state"]["files"] == ["result.txt"]
assert not any(name in sys.modules for name in blocked)
""",
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("outcome", ["completed", "failed", "interrupted"])
def test_completion_reader_preserves_optional_evidence_and_returns_fresh_snapshots(outcome):
    marker = {
        **_marker(),
        "outcome": outcome,
        "task_id": "task-1",
        "disposal_state": {"pending": ["resource-1"]},
        "extension": {"receipt": [1, 2]},
    }
    checkpoint = {reader.PENDING_COMPLETION_FINALIZATION_CHECKPOINT_KEY: marker}
    original = deepcopy(checkpoint)
    first = reader.pending_completion_finalization_from_checkpoint(checkpoint)
    assert first == marker
    first["binding_state"]["files"].clear()
    first["disposal_state"]["pending"].clear()
    first["extension"]["receipt"].clear()
    assert checkpoint == original
    marker["binding_state"]["files"].append({"path": "new.txt"})
    second = reader.pending_completion_finalization_from_checkpoint(checkpoint)
    assert second == marker and second is not first
    del checkpoint[reader.PENDING_COMPLETION_FINALIZATION_CHECKPOINT_KEY]
    assert reader.pending_completion_finalization_from_checkpoint(checkpoint) is None


def test_completion_reader_accepts_missing_marker_and_absent_optional_fields():
    for checkpoint in (None, {}, {reader.PENDING_COMPLETION_FINALIZATION_CHECKPOINT_KEY: None}):
        assert reader.pending_completion_finalization_from_checkpoint(checkpoint) is None
    for extra in ({}, {"task_id": None, "disposal_state": None}):
        marker = {**_marker(), **extra}
        assert (
            reader.pending_completion_finalization_from_checkpoint(
                {reader.PENDING_COMPLETION_FINALIZATION_CHECKPOINT_KEY: marker}
            )
            == marker
        )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("version", True, "unsupported format"),
        ("version", 2, "unsupported format"),
        ("version", "1", "unsupported format"),
        ("outcome", "unknown", "unsupported format"),
        ("environment_name", None, "environment_name must be a string"),
        ("environment_name", " coding", "whitespace"),
        ("binding_generation_id", 1, "binding_generation_id must be a string"),
        ("binding_generation_id", "", "cannot be blank"),
        ("execution_profile_fingerprint", [], "execution_profile_fingerprint must be a string"),
        ("execution_profile_fingerprint", " ", "cannot be blank"),
        ("task_id", 1, "task_id must be a string"),
        ("task_id", "", "cannot be blank"),
        ("disposal_state", [], "disposal state must be an object"),
        ("binding_state", [], "binding state must be an object"),
        ("binding_state", None, "binding state must be an object"),
    ],
)
def test_completion_reader_rejects_malformed_evidence_without_mutating_it(field, value, error):
    checkpoint = {
        reader.PENDING_COMPLETION_FINALIZATION_CHECKPOINT_KEY: {**_marker(), field: value}
    }
    original = deepcopy(checkpoint)
    with pytest.raises(ValueError, match=error):
        reader.pending_completion_finalization_from_checkpoint(checkpoint)
    assert checkpoint == original


def test_completion_reader_rejects_nonobject_and_missing_required_evidence():
    key = reader.PENDING_COMPLETION_FINALIZATION_CHECKPOINT_KEY
    for marker in ([], "marker", True):
        with pytest.raises(ValueError, match="checkpoint must be an object"):
            reader.pending_completion_finalization_from_checkpoint({key: marker})
    for field in _marker():
        marker = _marker()
        del marker[field]
        with pytest.raises(ValueError):
            reader.pending_completion_finalization_from_checkpoint({key: marker})
    marker = _marker()
    marker["binding_state"]["cycle"] = marker
    with pytest.raises(ValueError):
        reader.pending_completion_finalization_from_checkpoint({key: marker})


def test_completion_reader_enforces_the_exact_encoded_marker_byte_limit():
    marker = {**_marker(), "extension": "é"}
    limit = 4 * 1024 * 1024
    size = len(canonical_durable_json_bytes(marker, "marker"))
    marker["extension"] += "x" * (limit - size)
    assert len(canonical_durable_json_bytes(marker, "marker")) == limit
    checkpoint = {reader.PENDING_COMPLETION_FINALIZATION_CHECKPOINT_KEY: marker}
    assert reader.pending_completion_finalization_from_checkpoint(checkpoint) == marker
    marker["extension"] += "x"
    with pytest.raises(ValueError, match="checkpoint exceeds its byte limit"):
        reader.pending_completion_finalization_from_checkpoint(checkpoint)


def test_completion_reader_has_one_owner_and_preserves_the_public_session_key():
    from typing import get_type_hints

    from cayu.runtime import _environment_lifecycle as lifecycle
    from cayu.sessions import base

    assert base.PENDING_COMPLETION_FINALIZATION_CHECKPOINT_KEY is (
        reader.PENDING_COMPLETION_FINALIZATION_CHECKPOINT_KEY
    )
    assert lifecycle.completion_finalization is base.completion_finalization is reader
    assert not hasattr(lifecycle, "pending_completion_finalization_from_checkpoint")
    assert not hasattr(lifecycle, "_MAX_COMPLETION_FINALIZATION_CHECKPOINT_BYTES")
    assert get_type_hints(reader.pending_completion_finalization_from_checkpoint)


def test_environment_checkpoint_replacement_preserves_pending_completion_evidence():
    import asyncio

    from tests.core.test_environment_lifecycle import _lifecycle

    from cayu.sessions.base import InMemorySessionStore, RunRequest, SessionIdentity

    async def scenario():
        store = InMemorySessionStore()
        session = await store.create(
            RunRequest(agent_name="assistant", messages=[]),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        marker = _marker()
        key = reader.PENDING_COMPLETION_FINALIZATION_CHECKPOINT_KEY
        await store.checkpoint(session.id, {key: marker, "stale_context": True})
        await _lifecycle(store).checkpoint_preserving_runtime_state(
            session.id, {"context_compaction": {"summary": "bounded"}}
        )
        assert await store.load_checkpoint(session.id) == {
            key: marker,
            "context_compaction": {"summary": "bounded"},
        }

    asyncio.run(scenario())
