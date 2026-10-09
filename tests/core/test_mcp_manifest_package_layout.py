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
