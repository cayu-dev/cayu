"""Independent queue reconciliation and application stream ownership."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest
from tests.core.test_dispatch import _batch, _build, _create_resumable_session, _dispatch_request

import cayu
from cayu.events import EventType
from cayu.sessions.base import (
    InMemorySessionStore,
    _current_session_interaction_id,
    _current_session_run_epoch,
)
from cayu.storage.sqlite import SQLiteSessionStore


def test_queued_dispatch_reconciliation_composes_without_controllers(tmp_path: Path) -> None:
    """A standalone owner can replay and acknowledge real persisted authority."""

    path = tmp_path / "queued.sqlite"
    sessions = SQLiteSessionStore(path)
    h = _build([_batch("first answer"), _batch("queued answer")], session_store=sessions)
    _create_resumable_session(h.app, "queued-session")

    async def prepare():
        envelope = await h.app._prepare_queued_dispatch(
            _dispatch_request("queued-session", "queued-dispatch"), queue_task_id="queue-task"
        )
        events = [event async for event in h.app._dispatch_queued(envelope)]
        assert events[-1].type == EventType.SESSION_COMPLETED
        await sessions.close()
        return envelope

    envelope = asyncio.run(prepare())
    script = """
import asyncio
import importlib.abc
import sys

blocked = {
    "cayu.applications", "cayu.runtime._session_engine",
    "cayu.runtime._model_step_executor", "cayu.runtime._recovery_coordinator",
}

class RejectControllers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked:
            raise AssertionError(f"Queued dispatch imported {fullname}")

sys.meta_path.insert(0, RejectControllers())
from cayu.runtime._checkpoint_store import _RuntimeCheckpointSessionStore
from cayu.runtime._queued_dispatch_coordinator import QueuedDispatchCoordinator
from cayu.sessions.base import QueuedDispatchTerminalReceiptQuery
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tasks.dispatch import (
    DispatchStatus, _QueuedDispatchEnvelope, _QueuedDispatchSettlementState,
)
from cayu.vaults.redaction import SecretRedactor

def unexpected(*args, **kwargs):
    raise AssertionError("Terminal reconciliation must not execute or resolve providers")

async def resolve_session(session_id):
    assert session_id == "queued-session"
    return session_id, None

projected = []
async def project(event):
    projected.append(event.id)
    return event

async def run():
    store = SQLiteSessionStore(sys.argv[1])
    envelope = _QueuedDispatchEnvelope.model_validate_json(sys.stdin.read())
    coordinator = QueuedDispatchCoordinator(
        get_session_store=lambda: store,
        runtime_session_store=_RuntimeCheckpointSessionStore(store),
        engine=object(), subagents=object(), redactor=SecretRedactor(),
        resolve_session=resolve_session, get_agent=unexpected, get_provider=unexpected,
        get_environment=unexpected, redact_request=unexpected, project_event=project,
        load_model_completion=unexpected, run=unexpected, dispatch=unexpected,
    )
    try:
        query = QueuedDispatchTerminalReceiptQuery(limit=1)
        receipts = await coordinator.list_terminal_receipts(query)
        assert len(receipts) == 1
        assert receipts[0].queue_task_id == envelope.queue_task_id
        settlement = await coordinator.settlement_state(envelope)
        assert settlement.state is _QueuedDispatchSettlementState.TERMINAL_EVIDENCE_DURABLE
        assert settlement.terminal_status is DispatchStatus.COMPLETED
        events = [event async for event in coordinator.execute(envelope)]
        assert [event.id for event in events] == [envelope.terminal_event_id]
        assert projected == [envelope.terminal_event_id]
        try:
            await coordinator.acknowledge(envelope, dispatch_status=DispatchStatus.FAILED)
        except RuntimeError as error:
            assert "conflicts with its exact terminal event" in str(error)
        else:
            raise AssertionError("Mismatched queue outcome released terminal evidence")
        assert await coordinator.list_terminal_receipts(query) == receipts
        await coordinator.acknowledge(
            envelope, dispatch_status=DispatchStatus.COMPLETED, receipt=receipts[0],
        )
        assert await coordinator.list_terminal_receipts(query) == []
        await coordinator.acknowledge(envelope, dispatch_status=DispatchStatus.COMPLETED)
        assert [event.id async for event in coordinator.execute(envelope)] == [
            envelope.terminal_event_id,
        ]
        assert not blocked.intersection(sys.modules)
    finally:
        await store.close()

asyncio.run(run())
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        input=envelope.model_dump_json(),
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("signal_kind", ["error", "cancellation", "group"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_queued_dispatch_stream_preserves_caller_signal_during_cleanup(
    signal_kind: str, cleanup_fails: bool
) -> None:
    class FailingReleaseStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1
        terminal_interaction_publication_version = 1
        fail_release = False

        async def release_session_invocation(self, command):
            if self.fail_release:
                self.fail_release = False
                raise RuntimeError("injected invocation release failure")
            return await super().release_session_invocation(command)

    sessions = FailingReleaseStore()
    h = _build([_batch("first answer"), _batch("queued answer")], session_store=sessions)
    _create_resumable_session(h.app, "queued-signal")

    async def scenario() -> None:
        envelope = await h.app._prepare_queued_dispatch(
            _dispatch_request("queued-signal", "queued-dispatch"), queue_task_id="queue-task"
        )
        stream = h.app._dispatch_queued(envelope)
        signal = {
            "error": ValueError("consumer rejected event"),
            "cancellation": asyncio.CancelledError("caller cancelled"),
            "group": BaseExceptionGroup(
                "consumer failures",
                [ValueError("consumer failed"), asyncio.CancelledError("caller cancelled")],
            ),
        }[signal_kind]
        original_cause = LookupError("original consumer cause")
        signal.__cause__ = original_cause
        try:
            async for event in stream:
                if event.type == EventType.SESSION_RESUMED:
                    break
            else:
                pytest.fail("Queued dispatch did not enter a session invocation")
            sessions.fail_release = cleanup_fails
            with pytest.raises(type(signal)) as raised:
                await stream.athrow(signal)
            assert raised.value is signal
            if cleanup_fails:
                assert isinstance(signal.__cause__, BaseExceptionGroup)
                cleanup_failure, preserved_cause = signal.__cause__.exceptions
                assert type(cleanup_failure) is RuntimeError
                assert str(cleanup_failure) == "injected invocation release failure"
                assert preserved_cause is original_cause
                assert not sessions.fail_release
            else:
                assert signal.__cause__ is original_cause
        finally:
            await stream.aclose()
        assert _current_session_run_epoch("queued-signal") is None
        assert _current_session_interaction_id("queued-signal") is None
        assert h.app._admission.in_flight == 0

    asyncio.run(scenario())
