"""Transcript values and validation compose without session-store implementations."""

from __future__ import annotations

import importlib
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

import cayu


def _assert_store_independent(code: str, public_module: str) -> None:
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
            "cayu.sessions.base", "cayu.tasks.memory", "cayu.runtime.execution",
        }:
            raise AssertionError(f"Transcript contract imported {fullname}")
sys.meta_path.insert(0, RejectStores())
public = importlib.import_module(sys.argv[1])
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


def _assert_public_identity(owner: str, name: str) -> None:
    canonical = getattr(importlib.import_module(owner), name)
    for module in ("cayu", "cayu.sessions", "cayu.runtime", "cayu.sessions.base"):
        assert getattr(importlib.import_module(module), name) is canonical
    assert pickle.loads(f"ccayu.sessions.base\n{name}\n.".encode()) is canonical


def test_transcript_snapshot_public_imports_and_historical_pickle_global():
    _assert_public_identity("cayu.sessions.transcript_queries", "TranscriptSnapshot")


@pytest.mark.parametrize(
    "public_module", ("cayu", "cayu.sessions", "cayu.runtime", "cayu.sessions.transcript_queries")
)
def test_transcript_snapshot_and_fork_validation_without_stores(public_module):
    _assert_store_independent(
        """
from cayu.messages import Message, ToolCallPart
from cayu.sessions.records import TranscriptRecord
from cayu.sessions.transcript_queries import (
    fork_source_transcript_sha256, fork_transcript_is_accepted,
)
message = Message(role="assistant", content=[ToolCallPart(
    tool_call_id="call", tool_name="tool", arguments={"nested": {"value": 1}},
)])
snapshot = public.TranscriptSnapshot(records=[TranscriptRecord(index=2, message=message)], cursor=5)
assert snapshot.retained_position(2) == 0
assert snapshot.retained_position(5) == 1
try:
    snapshot.retained_position(3)
except ValueError:
    pass
else:
    raise AssertionError("Removed transcript position was accepted")
before = fork_source_transcript_sha256(snapshot)
def validate(messages, source):
    messages[0].content[0].arguments["nested"]["value"] = 2
    source.records.clear()
    return True
assert fork_transcript_is_accepted([message], snapshot, validate)
assert message.content[0].arguments["nested"]["value"] == 1
assert fork_source_transcript_sha256(snapshot) == before
assert not fork_transcript_is_accepted([message], snapshot, lambda messages, source: 1)
assert pickle.loads(pickle.dumps(snapshot)) == snapshot
assert get_type_hints(type(snapshot))["records"] == list[TranscriptRecord]
assert type(snapshot).model_json_schema()["title"] == "TranscriptSnapshot"
""",
        public_module,
    )


def test_bounded_transcript_text_without_stores():
    _assert_store_independent(
        """
from cayu.messages import Message, TextPart, ThinkingPart
from cayu.sessions.transcript_queries import (
    LATEST_TRANSCRIPT_TEXT_MAX_PARTS, TranscriptTextReadLimitExceeded,
    _bounded_transcript_message_text,
)
message = Message(role="assistant", content=[ThinkingPart(text="private"),
    TextPart(text="éclair"), TextPart(text=" and tea")])
assert _bounded_transcript_message_text(message, max_chars=6) == ("éclair", True)
assert _bounded_transcript_message_text(message, max_chars=100) == ("éclair and tea", False)
large = Message.model_construct(role="assistant",
    content=(ThinkingPart(text="private"),) * (LATEST_TRANSCRIPT_TEXT_MAX_PARTS + 1))
try:
    _bounded_transcript_message_text(large, max_chars=6)
except TranscriptTextReadLimitExceeded:
    pass
else:
    raise AssertionError("Content-part inspection bound was ignored")
""",
        "cayu.sessions.transcript_queries",
    )
