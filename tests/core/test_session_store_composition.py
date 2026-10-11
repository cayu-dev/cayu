"""Session contracts and store implementations have independent import boundaries."""

from __future__ import annotations

import importlib
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

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


@pytest.mark.parametrize(
    "public_module", ("cayu", "cayu.sessions", "cayu.runtime", "cayu.sessions.store")
)
def test_store_interface_and_custom_checkpoint_reader_compose_without_memory(public_module):
    _isolated(
        f"""
import asyncio
import importlib
from cayu.sessions.store import SessionStore

assert importlib.import_module({public_module!r}).SessionStore is SessionStore
async def unsupported(self, *args, **kwargs):
    raise NotImplementedError("Test backend only supplies checkpoint reads")
checkpoint = {{"checkpoint_schema_version": 13, "execution_snapshots": {{"cursor": 4}},
              "customer_state": {{"private": True}}}}
async def load_checkpoint(self, session_id):
    assert session_id == "session"
    return checkpoint
# Implement the existing abstract storage obligations without importing a
# concrete Cayu backend. The inherited derived read must use this backend.
methods = {{name: unsupported for name in SessionStore.__abstractmethods__}}
methods["load_checkpoint"] = load_checkpoint
CheckpointReader = type("CheckpointReader", (SessionStore,), methods)
store = CheckpointReader()
async def scenario():
    projected = await store.load_execution_snapshot_checkpoint("session")
    assert projected == {{"checkpoint_schema_version": 13, "execution_snapshots": {{"cursor": 4}}}}
    assert "customer_state" in checkpoint
    assert not store._supports_session_export_protocol()
    assert not store._supports_session_continuation_protocol()
asyncio.run(scenario())
"""
    )


def test_memory_store_composes_without_native_adapters_or_application():
    _isolated(
        """
import asyncio
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionIdentity
from cayu.sessions.requests import RunRequest

async def scenario():
    store = InMemorySessionStore()
    session = await store.create(RunRequest(agent_name="agent", messages=[]),
        identity=SessionIdentity(provider_name="provider", model="model"))
    await store.checkpoint(session.id, {"customer": {"revision": 1}})
    before = await store.load_checkpoint(session.id)
    def reject(session, checkpoint):
        checkpoint["customer"]["revision"] = 2
        raise ValueError("rejected transform")
    try:
        await store.transform_checkpoint(session.id, reject)
    except ValueError as error:
        assert str(error) == "rejected transform"
    else:
        raise AssertionError("Invalid transform was committed")
    assert await store.load_checkpoint(session.id) == before
    await store.transform_checkpoint(session.id,
        lambda session, checkpoint: {**checkpoint, "customer": {"revision": 3}})
    assert (await store.load_checkpoint(session.id))["customer"] == {"revision": 3}
asyncio.run(scenario())
""",
        allow_memory=True,
    )


@pytest.mark.parametrize(
    "name",
    (
        "SessionStore",
        "InMemorySessionStore",
        "SessionExecutionInProgress",
        "SessionMessageQueueStatus",
        "SessionRunFenced",
    ),
)
def test_store_public_imports_and_historical_symbol_pickles_keep_identity(name):
    canonical = getattr(importlib.import_module("cayu.sessions"), name)
    for module in ("cayu", "cayu.sessions", "cayu.runtime", "cayu.sessions.base"):
        assert getattr(importlib.import_module(module), name) is canonical
    historical = f"ccayu.sessions.base\n{name}\n.".encode()
    assert pickle.loads(historical) is canonical
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        assert pickle.loads(pickle.dumps(canonical, protocol)) is canonical
