"""Work-attempt admission and settlement compose without turn execution."""

from __future__ import annotations

import asyncio

import pytest
from tests.core.test_recovery_composition import _run_without_owners
from tests.core.test_work_attempt_admission import _configured_public_initial_admission

from cayu.events import EventType
from cayu.sessions._invocation_scope import (
    _current_session_interaction_id,
    _current_session_run_epoch,
    _deactivate_session_interaction,
    _deactivate_session_run_fence,
)
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionStatus
from cayu.storage.sqlite import SQLiteSessionStore, SQLiteTaskStore
from cayu.tasks.admission import WorkAttemptAdmissionState, WorkAttemptRecoveryRequired
from cayu.tasks.memory import InMemoryTaskStore
from cayu.vaults.redaction import SecretRedactor


def test_work_attempt_parts_import_without_application_or_session_engine() -> None:
    _run_without_owners(
        ("cayu.applications", "cayu.runtime._session_engine"),
        """
from cayu.runtime._session_preparation import SessionPreparation
from cayu.runtime._interaction_events import build_interaction_started_event
from cayu.runtime._work_attempt_execution import WorkAttemptExecution
from cayu.runtime._work_attempt_coordinator import WorkAttemptCoordinator
""",
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_admission_replay_and_pre_entry_release_need_no_turn_execution(
    backend, tmp_path, monkeypatch
) -> None:
    async def scenario() -> None:
        sessions = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "sessions.sqlite")
        )
        tasks = (
            InMemoryTaskStore()
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / "tasks.sqlite")
        )
        try:
            app, request, execution = await _configured_public_initial_admission(
                prefix="independent-admission",
                sessions=sessions,
                tasks=tasks,
                redactor=SecretRedactor(),
            )

            async def unexpected_execution(**kwargs):
                raise AssertionError("Admission and settlement cannot dispatch the turn loop.")
                yield  # Preserve the execution callable's async-generator contract.

            monkeypatch.setattr(
                app._work_attempt_execution, "_execute_session", unexpected_execution
            )
            monkeypatch.delattr(app, "_session_engine")
            admission = await app.admit_work_attempt(request, execution=execution)
            assert admission.state is WorkAttemptAdmissionState.ACTIVE
            assert admission.execution_entry is None
            events = await sessions.query_events(EventQuery(session_id=admission.session_id))
            assert [record.event.type for record in events].count(
                EventType.INTERACTION_STARTED
            ) == 1
            assert await app.admit_work_attempt(request, execution=execution) == admission
            assert (
                await sessions.query_events(EventQuery(session_id=admission.session_id)) == events
            )

            operations = app.work_attempt_execution
            assert await operations.settle_work_attempt_admission_handoff(admission)
            release = await operations.close_unentered_work_attempt_invocation(admission)
            assert release.interaction_id == admission.interaction_id
            assert release.profile_fingerprint == admission.source_execution_profile_fingerprint
            session = await sessions.load(admission.session_id)
            assert session is not None and session.status is SessionStatus.INTERRUPTED
            checkpoint = await sessions.load_checkpoint(admission.session_id)
            settled_events = await sessions.query_events(
                EventQuery(session_id=admission.session_id)
            )
            assert await operations.close_unentered_work_attempt_invocation(admission) == release
            assert await sessions.load_checkpoint(admission.session_id) == checkpoint
            assert (
                await sessions.query_events(EventQuery(session_id=admission.session_id))
                == settled_events
            )
            assert await tasks.load_work_attempt_admission(admission.admission_id) == admission
        finally:
            if backend == "sqlite":
                await tasks.close()
                await sessions.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("failure_type", [RuntimeError, asyncio.CancelledError])
@pytest.mark.parametrize("stage", ["factory", "iteration"])
def test_execution_failure_clears_local_ownership_and_retains_durable_entry(
    stage, failure_type, monkeypatch
) -> None:
    async def scenario() -> None:
        sessions = InMemorySessionStore()
        tasks = InMemoryTaskStore()
        app, request, execution = await _configured_public_initial_admission(
            prefix="failed-execution-factory",
            sessions=sessions,
            tasks=tasks,
            redactor=SecretRedactor(),
        )
        admission = await app.admit_work_attempt(request, execution=execution)
        session_id = admission.session_id
        checkpoint = await sessions.load_checkpoint(session_id)
        session = await sessions.load(session_id)
        operations = app.work_attempt_execution
        failure = failure_type("execution setup failed")
        calls = 0

        def fail_execution(**kwargs):
            nonlocal calls
            calls += 1
            assert _current_session_interaction_id(session_id) == admission.interaction_id
            assert _current_session_run_epoch(session_id) == kwargs["session"].run_epoch
            raise failure

        async def fail_iteration(**kwargs):
            fail_execution(**kwargs)
            yield  # Preserve the execution callable's async-generator contract.

        monkeypatch.setattr(
            app._work_attempt_execution,
            "_execute_session",
            fail_execution if stage == "factory" else fail_iteration,
        )
        monkeypatch.delattr(app, "_session_engine")
        # Advance the component directly, without an application's outer context guard.
        _deactivate_session_interaction(session_id)
        _deactivate_session_run_fence(session_id)
        stream = operations.execute_admitted_work_attempt(admission, lease_seconds=300)
        try:
            with pytest.raises(failure_type) as raised:
                await anext(stream)
            assert raised.value is failure
        finally:
            await stream.aclose()

        assert _current_session_interaction_id(session_id) is None
        assert _current_session_run_epoch(session_id) is None
        entered = await tasks.load_work_attempt_admission(admission.admission_id)
        assert entered is not None and entered.execution_entry is not None
        assert await sessions.load_checkpoint(session_id) == checkpoint
        assert await sessions.load(session_id) == session
        with pytest.raises(WorkAttemptRecoveryRequired, match="already entered"):
            async for _ in operations.execute_admitted_work_attempt(entered, lease_seconds=300):
                raise AssertionError("Failed execution cannot dispatch again without recovery.")
        assert calls == 1

    asyncio.run(scenario())


def test_preparation_keeps_its_composed_task_authority_store(monkeypatch) -> None:
    class RecordingTaskStore(InMemoryTaskStore):
        def __init__(self):
            super().__init__()
            self.reads = []

        async def load_task(self, task_id):
            self.reads.append(task_id)
            return await super().load_task(task_id)

    async def scenario() -> None:
        configured = RecordingTaskStore()
        replacement = RecordingTaskStore()
        app, request, _ = await _configured_public_initial_admission(
            prefix="captured-preparation",
            sessions=InMemorySessionStore(),
            tasks=configured,
            redactor=SecretRedactor(),
        )
        configured.reads.clear()
        app.task_store = replacement
        monkeypatch.delattr(app, "_session_engine")
        # The configured task is governed: ordinary preparation must reject it.
        # Looking through the mutable app attribute would instead see no task.
        assert await app._session_preparation.prepare_initial_run(request) is None
        assert configured.reads == [request.task_id]
        assert replacement.reads == []

    asyncio.run(scenario())
