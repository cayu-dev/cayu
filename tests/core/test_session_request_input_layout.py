"""Session input evidence is usable without importing session stores."""

from __future__ import annotations

import importlib
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

import cayu


@pytest.mark.parametrize(
    "name",
    (
        "SessionInputContractEvidence",
        "parse_session_input_contract_evidence",
        "session_input_messages_sha256",
        "system_prompt_messages_sha256",
        "session_messages_input_contract_evidence",
    ),
)
def test_input_evidence_historical_public_identity(name):
    owner = importlib.import_module("cayu.sessions.transcript_input")
    canonical = getattr(owner, name)
    assert getattr(importlib.import_module("cayu.sessions.base"), name) is canonical
    assert pickle.loads(f"ccayu.sessions.base\n{name}\n.".encode()) is canonical
    for module in ("cayu", "cayu.sessions", "cayu.runtime"):
        exports = importlib.import_module(module + "._exports").EXPORTS
        if name in exports:
            assert exports[name] == ("cayu.sessions.transcript_input", name)
            assert getattr(importlib.import_module(module), name) is canonical


def test_input_evidence_composes_without_store_or_runtime_execution():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import pickle
import sys
from typing import get_type_hints

class RejectImplementations(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("cayu.storage") or fullname in {
            "cayu.sessions.base", "cayu.tasks.memory",
        } or (fullname.startswith("cayu.runtime.") and fullname not in {
            "cayu.runtime._exports", "cayu.runtime.execution_identity",
        }):
            raise AssertionError(f"Input evidence imported {fullname}")
sys.meta_path.insert(0, RejectImplementations())
from cayu.messages import Message
from cayu.sessions import transcript_input as inputs
from cayu.sessions._model_failover import MODEL_TARGET_PROJECTION_METADATA_KEY
from cayu.sessions.forks import FORK_SOURCE_SNAPSHOT_METADATA_KEY

messages = [Message.text("system", "policy"), Message.text("user", "input")]
marker = inputs.session_messages_input_contract_evidence(
    messages, message_start_index=2, redactions_applied=True,
    structured_output_requested=False,
)
evidence = inputs.parse_session_input_contract_evidence(marker)
assert (evidence.message_start_index, evidence.message_count) == (2, 2)
assert evidence.redactions_applied and not evidence.structured_output_requested
assert evidence.messages_sha256 == inputs.session_input_messages_sha256(messages)
assert pickle.loads(pickle.dumps(evidence)) == evidence
assert get_type_hints(type(evidence))
changed = [messages[0], Message.text("user", "changed input")]
assert inputs.session_input_messages_sha256(changed) != evidence.messages_sha256
assert inputs.system_prompt_messages_sha256(changed) == inputs.system_prompt_messages_sha256(messages)
detached = inputs._copy_caller_input_message(messages[1])
assert detached == messages[1] and detached is not messages[1]
assert detached.content[0] is not messages[1].content[0]
for invalid in (marker.replace("v1:2:", "v1:02:"), marker + ":extra", marker.upper()):
    try:
        inputs.parse_session_input_contract_evidence(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("Noncanonical input marker was accepted")
assert MODEL_TARGET_PROJECTION_METADATA_KEY == "cayu:model_target_projection"
assert FORK_SOURCE_SNAPSHOT_METADATA_KEY == "cayu:fork_source_snapshot"
""",
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
