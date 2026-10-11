"""Session contracts and store implementations have independent import boundaries."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import cayu


def _isolated(code: str, *, allow_memory: bool = False) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys

class RejectConcreteOwners(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        forbidden = {
            "cayu.sessions.base", "cayu.storage.sqlite", "cayu.storage.postgres",
            "cayu.applications", "cayu.runtime._session_engine",
        }
        if sys.argv[1] == "False":
            forbidden.add("cayu.sessions.memory")
        if fullname in forbidden:
            raise AssertionError(f"Unexpected concrete owner: {fullname}")
sys.meta_path.insert(0, RejectConcreteOwners())
"""
            + code,
            str(allow_memory),
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_shared_store_publication_callbacks_work_without_store_implementations():
    _isolated(
        """
import asyncio
from cayu.sessions._runtime_publication import (
    _OwnedOffThreadSessionCommitGuard,
    _prepare_initial_session_operation_records,
    _run_session_commit_guard_owned,
)
from cayu.sessions.records import Session
from cayu.sessions.invocation import SessionInvocation

session = Session(id="session", agent_name="agent", provider_name="provider", model="model",
                  invocation=SessionInvocation(origin={"trust": "unattributed"},
                      root_invocation_id="c092a88f-bc0a-4ec0-a92c-d3b8cc21d8cb", source="sdk_run", root_session_id="session"),
                  metadata={"customer": {"value": 1}})
def initialize(detached):
    detached.metadata["customer"]["value"] = 2
    return {"customer_operation": {"ready": True}}
records = _prepare_initial_session_operation_records(session, initialize)
assert records == {"customer_operation": {"ready": True}}
assert session.metadata == {"customer": {"value": 1}}
assert _prepare_initial_session_operation_records(session, None) == {}

calls = []
async def scenario():
    assert await _run_session_commit_guard_owned(lambda: calls.append("inline")) is None
    assert await _run_session_commit_guard_owned(
        _OwnedOffThreadSessionCommitGuard(lambda: calls.append("thread"))) is None
asyncio.run(scenario())
assert calls == ["inline", "thread"]
"""
    )


def test_interruption_marker_projection_works_without_store_implementations():
    _isolated(
        """
from cayu.sessions.recovery import (
    _project_interruption_cascade_marker,
    _project_interruption_cascade_marker_fields,
)

marker = {"attempt_id": "attempt", "interrupt_payload": {"large": "x" * 10000},
          "generation": 2, "claim_id": "claim", "failure_recorded": False}
projection = _project_interruption_cascade_marker(marker)
assert projection == {"attempt_id": "attempt", "interrupt_payload": {},
                      "generation": 2, "claim_id": "claim", "failure_recorded": False}
assert marker["interrupt_payload"] == {"large": "x" * 10000}
assert _project_interruption_cascade_marker(None) is None
assert _project_interruption_cascade_marker([]) == {}
assert _project_interruption_cascade_marker_fields("null", {}, {}) is None
assert _project_interruption_cascade_marker({"attempt_id": "x" * 129})["attempt_id"] is None
"""
    )
