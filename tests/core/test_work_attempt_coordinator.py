"""Work-attempt composition and delegated-stream lifecycle contracts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import cayu


def test_work_attempt_coordinator_composes_without_application_controllers() -> None:
    script = """
import asyncio
import importlib.abc
import os
import sys

blocked = {
    "cayu.applications", "cayu.runtime._session_engine",
    "cayu.runtime._model_step_executor", "cayu.runtime._recovery_coordinator",
}

class RejectControllers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked:
            raise AssertionError(f"Work-attempt coordinator imported {fullname}")

sys.meta_path.insert(0, RejectControllers())
from cayu.messages import Message
from cayu.runtime._work_attempt_coordinator import WorkAttemptCoordinator
from cayu.sessions.base import InMemorySessionStore, ResumeRequest, RunRequest
from cayu.tasks.admission import WorkAttemptExecutionRequest
from cayu.tasks.base import InMemoryTaskStore
from cayu.vaults.redaction import SecretRedactor

class ReachedEngine(Exception):
    pass

class Engine:
    def __init__(self):
        self.admissions = []

    def work_attempt_source_request_sha256(self, request, *, kind):
        return "0" * 64

    def work_attempt_source_snapshot(self, request, *, kind, source_request_sha256):
        assert source_request_sha256 == "0" * 64
        return None

    async def admit_initial_work_attempt(self, request, *, authority):
        assert request.max_steps == 3
        self.admissions.append((request, authority))
        request.messages.append(Message.text("user", "engine-owned input"))
        raise ReachedEngine

    async def admit_continuation_work_attempt(self, request, *, authority):
        assert request.session_id == "private-session"
        self.admissions.append((request, authority))
        raise ReachedEngine

async def resolve_session(value):
    assert value == "public-session"
    return "private-session", "private-session"

def checkpoint_guard(*args, **kwargs):
    raise AssertionError("Admission does not perform session recovery")

async def run():
    store = [InMemoryTaskStore()]
    engine = Engine()
    coordinator = WorkAttemptCoordinator(
        get_task_store=lambda: store[0], session_store=InMemorySessionStore(),
        engine=engine, redactor=SecretRedactor(),
        apply_run_defaults=lambda request: request.model_copy(update={"max_steps": 3}),
        resolve_session=resolve_session, checkpoint_guard=checkpoint_guard,
    )
    execution = WorkAttemptExecutionRequest(
        admission_id="admission", claim_id="claim", attempt_id="attempt",
        interaction_id="interaction", worker_id="worker", generation=1, lease_seconds=30,
    )
    initial = RunRequest(
        agent_name="worker", session_id="initial-session", task_id="task",
        messages=[Message.text("user", "caller-owned input")],
    )
    continuation = ResumeRequest(
        session_id="public-session", messages=[Message.text("user", "continue")],
    )
    owner = coordinator.current_execution_owner_id()
    for request, selection in (
        (initial, execution),
        (continuation, execution.model_copy(update={
            "task_id": "task", "predecessor_admission_id": "previous",
        })),
    ):
        try:
            await coordinator.admit(request, execution=selection)
        except ReachedEngine:
            pass
        else:
            raise AssertionError("Admission did not reach the supplied engine")
    assert len(initial.messages) == 1
    assert continuation.session_id == "public-session"
    assert [authority.kind for _, authority in engine.admissions] == ["initial", "continuation"]
    for copied, authority in engine.admissions:
        assert authority.execution_owner_id == owner
        assert authority.source_request_sha256 == "0" * 64
        assert authority.request is not execution
    assert coordinator.current_execution_owner_id() == owner
    previous_getpid = os.getpid
    try:
        os.getpid = lambda: previous_getpid() + 1
        replacement = coordinator.current_execution_owner_id()
        assert replacement != owner
        assert replacement.rsplit(":", 1)[-1] != owner.rsplit(":", 1)[-1]
        assert coordinator.current_execution_owner_id() == replacement
    finally:
        os.getpid = previous_getpid
    store[0] = None
    try:
        await coordinator.admit(initial, execution=execution)
    except RuntimeError as error:
        assert str(error) == "task_store is required for work-attempt execution."
    else:
        raise AssertionError("Coordinator used a stale TaskStore")
    assert len(engine.admissions) == 2
    assert not blocked.intersection(sys.modules)

asyncio.run(run())
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("close_early", [False, True])
def test_work_attempt_stream_clears_inherited_authority_before_yield(close_early: bool) -> None:
    import asyncio

    from tests.core.test_work_attempt_admission import _configured_public_initial_admission

    from cayu.sessions.base import (
        InMemorySessionStore,
        _current_session_interaction_id,
        _current_session_run_epoch,
    )
    from cayu.tasks.admission import WorkAttemptRunRequest
    from cayu.tasks.base import InMemoryTaskStore
    from cayu.vaults.redaction import SecretRedactor

    async def scenario() -> None:
        app, source, execution = await _configured_public_initial_admission(
            prefix="caller-context",
            sessions=InMemorySessionStore(),
            tasks=InMemoryTaskStore(),
            redactor=SecretRedactor(),
        )
        admitted = await app.admit_work_attempt(source, execution=execution)
        assert _current_session_run_epoch(admitted.session_id) == 1
        assert _current_session_interaction_id(admitted.session_id) == admitted.interaction_id
        stream = app._execute_work_attempt(
            WorkAttemptRunRequest(
                admission_id=admitted.admission_id,
                claim_id=admitted.claim.claim_id,
                worker_id=admitted.claim.worker_id,
                generation=admitted.claim.generation,
                lease_seconds=300,
            )
        )
        try:
            async for _ in stream:
                assert _current_session_run_epoch(admitted.session_id) is None
                assert _current_session_interaction_id(admitted.session_id) is None
                if close_early:
                    break
        finally:
            await stream.aclose()
        assert _current_session_run_epoch(admitted.session_id) is None
        assert _current_session_interaction_id(admitted.session_id) is None
        assert app._admission.in_flight == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("signal_kind", ["error", "cancellation", "group"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_work_attempt_stream_preserves_caller_signal_during_cleanup(
    signal_kind: str, cleanup_fails: bool
) -> None:
    import asyncio

    from tests.core.test_work_attempt_admission import _configured_public_initial_admission

    from cayu.sessions.base import (
        InMemorySessionStore,
        _current_session_interaction_id,
        _current_session_run_epoch,
    )
    from cayu.tasks.admission import WorkAttemptRunRequest
    from cayu.tasks.base import InMemoryTaskStore
    from cayu.vaults.redaction import SecretRedactor

    class FailingReleaseStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1
        terminal_interaction_publication_version = 1
        fail_release = False

        async def release_session_invocation(self, command):
            if self.fail_release:
                self.fail_release = False
                raise RuntimeError("injected invocation release failure")
            return await super().release_session_invocation(command)

    async def scenario() -> None:
        sessions = FailingReleaseStore()
        app, source, execution = await _configured_public_initial_admission(
            prefix="caller-signal",
            sessions=sessions,
            tasks=InMemoryTaskStore(),
            redactor=SecretRedactor(),
        )
        admitted = await app.admit_work_attempt(source, execution=execution)
        stream = app._execute_work_attempt(
            WorkAttemptRunRequest(
                admission_id=admitted.admission_id,
                claim_id=admitted.claim.claim_id,
                worker_id=admitted.claim.worker_id,
                generation=admitted.claim.generation,
                lease_seconds=300,
            )
        )
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
            await anext(stream)
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
        assert _current_session_run_epoch(admitted.session_id) is None
        assert _current_session_interaction_id(admitted.session_id) is None
        assert app._admission.in_flight == 0

    asyncio.run(scenario())
