"""Recovery ownership stays composable across the execution boundary."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import cayu


def _run_without_owners(blocked: tuple[str, ...], operation: str) -> None:
    script = (
        """
import importlib.abc
import sys

blocked = set(sys.argv[1:])
class RejectOwners(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked:
            raise AssertionError(f"Independent component imported {fullname}")
sys.meta_path.insert(0, RejectOwners())
"""
        + operation
        + """
assert not blocked.intersection(sys.modules)
"""
    )
    result = subprocess.run(
        [sys.executable, "-c", script, *blocked],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_recovery_parts_work_without_application_or_execution():
    _run_without_owners(
        (
            "cayu.applications",
            "cayu.runtime._session_engine",
            "cayu.runtime._session_recovery",
            "cayu.runtime._recovery_coordinator",
            "cayu.runtime._incomplete_session_recovery",
        ),
        """
import asyncio
from cayu.runtime._pending_tool_round_recovery import PendingToolRoundRecovery
from cayu.runtime._workspace_observation_recovery import WorkspaceObservationRecovery
from cayu.runtime._recovery_ownership import RecoveryOwnership
from cayu.runtime._terminal_evidence_reader import TerminalEvidenceReader
from cayu.runtime._terminal_evidence_finalization import TerminalEvidenceFinalization
from cayu.runtime._terminal_event_publication import TerminalEventPublication
from cayu.runtime._session_finalization import SessionFinalization
from cayu.runtime._recovery_admission import RecoveryAdmission
from cayu.runtime._work_attempt_coordinator import WorkAttemptCoordinator
from cayu.runtime._durable_tool_round import DeferredInteractionInput
from cayu.sessions.base import InMemorySessionStore, RunRequest
from cayu.sessions.records import SessionIdentity
from cayu.messages import Message
from cayu.vaults.redaction import SecretRedactor

async def scenario():
    store = InMemorySessionStore()
    session = await store.create(
        RunRequest(agent_name="independent", messages=[Message.text("user", "hello")]),
        identity=SessionIdentity(provider_name="fake", model="fake-model"),
    )
    deferred = DeferredInteractionInput(store, SecretRedactor())
    before = await store.load_checkpoint(session.id)
    assert not await deferred.materialize_if_present(session.id)
    assert await store.load_checkpoint(session.id) == before
asyncio.run(scenario())
""",
    )


def test_execution_imports_without_continuation_recovery():
    _run_without_owners(
        (
            "cayu.applications",
            "cayu.runtime._session_recovery",
            "cayu.runtime._recovery_coordinator",
            "cayu.runtime._incomplete_session_recovery",
            "cayu.runtime._provider_disposition_recovery",
        ),
        """
from cayu.runtime._session_engine import SessionEngine
from cayu.runtime._work_attempt_coordinator import WorkAttemptEngine
assert callable(SessionEngine.prepare_resume)
assert callable(SessionEngine.resume)
assert callable(SessionEngine.continue_run)
""",
    )
