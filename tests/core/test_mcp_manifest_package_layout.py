"""Manifest history contracts compose without session-store implementations."""

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
from datetime import UTC, datetime
from typing import get_type_hints

class RejectStores(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("cayu.storage") or fullname in {
            "cayu.sessions.base", "cayu.tasks.memory",
        } or (fullname.startswith("cayu.runtime.") and fullname != "cayu.runtime._exports"):
            raise AssertionError(f"Manifest contract imported {fullname}")
sys.meta_path.insert(0, RejectStores())
public = importlib.import_module(sys.argv[1])
from cayu.sessions import mcp_manifest_history as history

source = "sha256:" + "1" * 64
server = "sha256:" + "2" * 64
tools = [{"tool_id": "sha256:" + "3" * 64, "contract_hash": "sha256:" + "4" * 64}]
baseline = public.McpManifestBaseline(
    history_key="sha256:" + "5" * 64, generation=1,
    manifest_identity="sha256:" + "6" * 64,
    manifest_hash=history._mcp_authoritative_manifest_hash(
        source_manifest_hash=source, server_hash=server, tools=tools, exposed_tools=tools,
    ),
    source_manifest_hash=source, server_hash=server, tools=tools, exposed_tools=tools,
    accepted_session_ref=history._mcp_manifest_session_ref("session"),
    accepted_event_id="accepted-event", accepted_at=datetime(2026, 1, 1, tzinfo=UTC),
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
        "McpManifestBaseline",
        "McpManifestBaselineLoadResult",
        "McpManifestPublicationResult",
        "McpManifestHistoryConflict",
    ),
)
def test_manifest_public_imports_and_historical_pickle_globals(name):
    canonical = getattr(importlib.import_module("cayu.sessions.mcp_manifest_history"), name)
    for module in ("cayu", "cayu.sessions", "cayu.runtime", "cayu.sessions.base"):
        assert getattr(importlib.import_module(module), name) is canonical
    assert pickle.loads(f"ccayu.sessions.base\n{name}\n.".encode()) is canonical


@pytest.mark.parametrize(
    "public_module", ("cayu", "cayu.sessions", "cayu.runtime", "cayu.sessions.mcp_manifest_history")
)
def test_manifest_values_and_hash_evidence_without_stores(public_module):
    _without_stores(
        """
assert type(baseline) is history.McpManifestBaseline
loaded = public.McpManifestBaselineLoadResult(baselines={baseline.history_key: baseline})
published = public.McpManifestPublicationResult(
    published=True, baselines={baseline.history_key: baseline},
)
tools[0]["contract_hash"] = "sha256:" + "7" * 64
assert baseline.tools[0]["contract_hash"] == "sha256:" + "4" * 64
baseline.tools[0]["contract_hash"] = "sha256:" + "8" * 64
for result in (loaded, published):
    assert result.baselines[baseline.history_key].tools[0]["contract_hash"] == "sha256:" + "4" * 64
    assert pickle.loads(pickle.dumps(result)) == result
    assert get_type_hints(type(result))["baselines"] == dict[str, history.McpManifestBaseline]
    assert type(result).model_json_schema()["$defs"]["McpManifestBaseline"]["title"] == "McpManifestBaseline"
try:
    public.McpManifestBaselineLoadResult(baselines={baseline.history_key: baseline})
except ValueError as error:
    assert "authoritative MCP manifest evidence" in str(error)
else:
    raise AssertionError("Mutated hash evidence was accepted")
error = public.McpManifestHistoryConflict("history changed")
assert type(pickle.loads(pickle.dumps(error))) is type(error)
""",
        public_module,
    )


@pytest.mark.parametrize("encoding", ("object", "json"))
def test_stored_manifest_decoding_and_redaction_without_stores(encoding):
    _without_stores(
        f"encoding = {encoding!r}\n"
        + """
import json
import traceback

decode = (history._stored_mcp_manifest_baseline_json if encoding == "json"
          else history._stored_mcp_manifest_baseline)
def encode(value):
    return json.dumps(value) if encoding == "json" else value
payload = baseline.model_dump(mode="json")
restored = decode(baseline.history_key, 1, encode(payload))
assert restored == baseline
payload["tools"][0]["contract_hash"] = "sensitive-invalid-evidence"
assert restored.tools == baseline.tools
for key, generation, value in (
    (baseline.history_key, 1, payload),
    (baseline.history_key, True, baseline.model_dump(mode="json")),
    ("sha256:" + "9" * 64, 1, baseline.model_dump(mode="json")),
):
    try:
        decode(key, generation, encode(value))
    except history._McpManifestBaselineEvidenceInvalid as error:
        assert str(error) == "Stored MCP manifest baseline evidence is invalid."
        assert "sensitive-invalid-evidence" not in "".join(traceback.format_exception(error))
    else:
        raise AssertionError("Invalid stored evidence was accepted")
keys = (baseline.history_key,)
assert history._validate_mcp_manifest_history_keys(keys) == keys
try:
    history._validate_mcp_manifest_history_keys(keys + keys)
except ValueError:
    pass
else:
    raise AssertionError("Duplicate history keys were accepted")
""",
        "cayu.sessions.mcp_manifest_history",
    )


def test_manifest_publication_state_validation_without_stores():
    _without_stores(
        """
from cayu.events import Event, EventType

key = baseline.history_key
payload = {field: getattr(baseline, field) for field in (
    "history_key", "manifest_identity", "manifest_hash", "source_manifest_hash", "server_hash",
)}
payload.update(status="first_seen", outcome="accepted")
event = Event(id=baseline.accepted_event_id, session_id="session",
    timestamp=baseline.accepted_at, type=EventType.MCP_MANIFEST_CHECKED, payload=payload)
validate = history._validate_mcp_manifest_publication_state
validate(expected_generations={key: None}, current_baselines={},
    baseline_updates={key: baseline}, events=[event])
unchanged = event.model_copy(update={"payload": {**payload, "status": "unchanged"}})
validate(expected_generations={key: 1}, current_baselines={key: baseline},
    baseline_updates={}, events=[unchanged])
for updates, events in (
    ({}, [event]),
    ({key: baseline}, [unchanged]),
    ({}, [unchanged, unchanged]),
    ({}, [event.model_copy(update={"payload": {**payload, "status": "changed"}})]),
):
    try:
        validate(expected_generations={key: 1}, current_baselines={key: baseline},
            baseline_updates=updates, events=events)
    except ValueError:
        pass
    else:
        raise AssertionError("Inconsistent manifest publication was accepted")
assert event.payload == payload
assert baseline.generation == 1
""",
        "cayu.sessions.mcp_manifest_history",
    )
