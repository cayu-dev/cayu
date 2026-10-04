"""Checkpoint decoding composes without runtime recovery or execution owners."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import get_type_hints

import cayu
from cayu.sessions import _pending_tool_round_reader as reader


def test_pending_round_reader_composes_without_runtime_execution():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import asyncio
import sys

blocked = (
    "cayu.runtime._tool_round_recovery", "cayu.runtime._checkpoint_redaction",
    "cayu.runtime._shared_artifact_results", "cayu.runtime._web_access_results",
    "cayu.runtime._approval_support", "cayu.runtime._runtime_records",
    "cayu.runtime._session_engine", "cayu.runtime._recovery_coordinator",
)
class BlockExecution:
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            raise AssertionError(fullname)
sys.meta_path.insert(0, BlockExecution())

from cayu.sessions._pending_tool_round import PendingToolRound
from cayu.sessions import _pending_tool_round_reader as reader
from cayu.vaults.redaction import SecretRedactor

record = PendingToolRound(
    model_step_id="mstep_" + "1" * 32,
    model_attempt_id="matt_" + "2" * 32,
    tool_round_id="tround_" + "3" * 32,
    agent_name="worker",
    tool_calls=[{"tool_call_id": "call-1", "tool_name": "echo",
                 "arguments": {"text": "first"}}],
)
class Store:
    def __init__(self):
        self.checkpoint = {"pending_tool_round": record.model_dump(mode="json")}
        self.reads = 0
    async def load_checkpoint(self, session_id):
        assert session_id == "session"
        self.reads += 1
        return self.checkpoint

async def run():
    store = Store()
    source, first = await reader.load_pending_tool_round(store, "session")
    assert source is store.checkpoint and first == record
    source["pending_tool_round"]["tool_calls"][0]["arguments"]["text"] = "second"
    _, second = await reader.load_pending_tool_round(store, "session")
    assert first.tool_calls[0].arguments["text"] == "first"
    assert second.tool_calls[0].arguments["text"] == "second"
    assert store.reads == 2
    secret = "reader-boundary-private-value"
    source["pending_tool_round"]["tool_calls"][0]["arguments"]["text"] = secret
    redactor = SecretRedactor().with_secret(secret)
    for consume in (False, True):
        try:
            await reader.load_pending_tool_round(
                store, "session", redactor=redactor, consume_on_rejection=consume,
            )
        except ValueError as error:
            assert secret not in str(error)
        else:
            raise AssertionError("Secret-bearing round was admitted")
        assert bool(source) is not consume
    assert store.reads == 4
    assert not any(name in sys.modules for name in blocked)

asyncio.run(run())
""",
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_pending_round_reader_and_runtime_share_canonical_validation():
    from cayu.runtime import _tool_round_recovery as recovery
    from cayu.sessions import _checkpoint_secret_validation as secrets
    from cayu.sessions import _pending_tool_round as records
    from cayu.tools import _shared_artifact_results, _web_access_results

    assert recovery.pending_round_reader is reader
    assert reader.pending_rounds is records
    assert reader.durable_value_contains_secret is secrets.durable_value_contains_secret
    assert (
        secrets.persisted_web_access_control_paths
        is _web_access_results.persisted_web_access_control_paths
    )
    assert (
        secrets.persisted_shared_artifact_control_paths
        is _shared_artifact_results.persisted_shared_artifact_control_paths
    )
    for name in (
        "load_pending_tool_round",
        "pending_tool_round_from_checkpoint",
        "_pending_tool_round_from_owned_checkpoint",
        "_require_executable_pending_tool_round",
    ):
        assert not hasattr(recovery, name)
        function = getattr(reader, name)
        assert function.__module__ == reader.__name__
        get_type_hints(function)
    for name in ("_checkpoint_redaction", "_web_access_results", "_shared_artifact_results"):
        assert importlib.util.find_spec("cayu.runtime." + name) is None
