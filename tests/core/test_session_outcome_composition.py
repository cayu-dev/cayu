"""Session outcomes compose below ordinary turn scheduling."""

from __future__ import annotations

import asyncio

import pytest
from tests.core.test_recovery_composition import _run_without_owners

from cayu import AgentSpec, CayuApp, RunRequest
from cayu.sessions.base import InMemorySessionStore
from cayu.sessions.records import SessionIdentity
from cayu.sessions.requests import run_request_with_task_invocation
from cayu.storage.sqlite import SQLiteSessionStore, SQLiteTaskStore
from cayu.tasks.creation import TaskCreate
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.records import TaskStatus


def test_session_task_lifecycle_imports_without_application_or_engine() -> None:
    _run_without_owners(
        ("cayu.applications", "cayu.runtime._session_engine"),
        "from cayu.runtime._session_task_lifecycle import SessionTaskLifecycle\n",
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("worker_owned", [False, True])
def test_session_task_lifecycle_attaches_and_completes_without_engine(
    backend, worker_owned, tmp_path, monkeypatch
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
            app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
            app.register_agent(AgentSpec(name="task-agent", model="fake-model"))
            lifecycle = app._session_tasks
            registered_agent = app._get_registered_agent("task-agent")
            monkeypatch.delattr(app, "_session_engine")
            # Replacing the public attribute does not replace composed task authority.
            app.task_store = InMemoryTaskStore()
            task = await tasks.create_task(TaskCreate(task_id="independent-task", type="job"))
            if worker_owned:
                task = await tasks.claim_task("worker", lease_seconds=300)
                assert task is not None
            snapshot = await tasks.load_invocation_snapshot(task.id)
            assert snapshot is not None
            request = run_request_with_task_invocation(
                RunRequest(agent_name="task-agent", task_id=task.id, messages=[]), snapshot
            )
            session = await sessions.create(
                request, identity=SessionIdentity(provider_name="fake", model="fake-model")
            )
            attached = await lifecycle.start_task(
                task_id=task.id,
                session=session,
                worker_id=task.worker_id,
                lease_expires_at=task.lease_expires_at,
                handoff_id=task.interrupted_handoff_id,
            )
            assert attached.status is TaskStatus.RUNNING
            assert attached.session_instance_id == session.instance_id
            assert await lifecycle.linked_continuation_task_id(
                session_id=session.id,
                task_worker_id=task.worker_id,
                task_handoff_id=task.interrupted_handoff_id,
            ) == (task.id, session.instance_id)
            completed = await lifecycle.complete_task(
                task_id=task.id,
                task_worker_id=task.worker_id,
                task_handoff_id=task.interrupted_handoff_id,
                session=session,
                registered_agent=registered_agent,
                registered_environment=None,
            )
            assert completed.status is TaskStatus.COMPLETED
            assert completed.result == {
                "session_id": session.id,
                "agent_name": "task-agent",
                "environment_name": None,
            }
            assert await tasks.load_task(task.id) == completed
            assert await app.task_store.load_task(task.id) is None
        finally:
            if backend == "sqlite":
                await tasks.close()
                await sessions.close()

    asyncio.run(scenario())


def test_session_outcome_publication_imports_without_application_or_engine() -> None:
    _run_without_owners(
        ("cayu.applications", "cayu.runtime._session_engine"),
        "from cayu.runtime._session_outcome_publication import SessionOutcomePublication\n",
    )


@pytest.mark.parametrize("outcome", ["completed", "failed", "interrupted"])
def test_governed_outcome_publication_works_without_turn_execution(outcome, monkeypatch) -> None:
    import time

    from tests.core.test_work_attempt_admission import _configured_public_initial_admission

    from cayu.events import EventType
    from cayu.runtime._session_outcome_publication import SessionOutcomeContext, SessionOutcomeState
    from cayu.runtime.loop_policies import BeforeStopDecision
    from cayu.sessions._invocation_scope import (
        _activate_session_interaction,
        _activate_session_run_fence,
        _deactivate_session_interaction,
        _deactivate_session_run_fence,
    )
    from cayu.sessions.event_queries import EventQuery
    from cayu.sessions.records import SessionStatus
    from cayu.vaults.redaction import SecretRedactor

    async def scenario() -> None:
        sessions = InMemorySessionStore()
        tasks = InMemoryTaskStore()
        app, request, execution = await _configured_public_initial_admission(
            prefix="independent-outcome",
            sessions=sessions,
            tasks=tasks,
            redactor=SecretRedactor(),
        )
        monkeypatch.delattr(app, "_session_engine")
        admission = await app.admit_work_attempt(request, execution=execution)
        session = await sessions.load(admission.session_id)
        assert session is not None
        invocation = await app._work_attempt_execution._enter_work_attempt_invocation_context(
            admission,
            session=session,
            checkpoint=await sessions.load_checkpoint(session.id),
            lease_seconds=300,
        )
        _activate_session_run_fence(session)
        _activate_session_interaction(session.id, admission.interaction_id)
        try:

            async def materialize() -> None:
                await sessions.replace_initial_transcript_messages(
                    session.id, request.messages, request.messages
                )

            state = SessionOutcomeState(session=session)
            context = SessionOutcomeContext(
                state=state,
                registered_agent=invocation.registered_agent,
                registered_environment=invocation.registered_environment,
                execution_profile=invocation.profile,
                invocation_context=invocation,
                environment_name=session.environment_name,
                run_started_at=time.monotonic(),
                turn_usage_tracker=app._run_limit_controller.usage_tracker(session.id),
                active_run=None,
                task_id=admission.task_id,
                task_worker_id=admission.claim.worker_id,
                task_lease_expires_at=admission.claim.lease_expires_at,
                task_handoff_id=None,
                request_loop_policies=invocation.request_loop_policies,
            )
            owner = app._session_outcomes
            if outcome == "failed":
                stream = owner.fail(context, RuntimeError("independent failure"), materialize)
            else:
                await materialize()
                if outcome == "completed":
                    (
                        completed,
                        _,
                        _,
                    ) = await app._session_finalization.publish_interaction_transition(
                        session=session,
                        invocation_context=invocation,
                        agent_name=invocation.registered_agent.spec.name,
                        environment_name=session.environment_name,
                        to_status=SessionStatus.COMPLETED,
                        execution_profile=invocation.profile,
                    )
                    stream = owner.complete_session(context, completed, None, None)
                else:
                    stream = owner.publish_stop_interruption(
                        context, BeforeStopDecision.interrupt("independent stop")
                    )
            try:
                events = [event async for event in stream]
            finally:
                await stream.aclose()
            terminal = {
                "completed": EventType.SESSION_COMPLETED,
                "failed": EventType.SESSION_FAILED,
                "interrupted": EventType.SESSION_INTERRUPTED,
            }[outcome]
            terminal_index = next(i for i, event in enumerate(events) if event.type is terminal)
            turn_index = next(
                i for i, event in enumerate(events) if event.type is EventType.TURN_COMPLETED
            )
            assert turn_index < terminal_index
            assert sum(event.type is terminal for event in events) == 1
            retained = await sessions.load(session.id)
            assert retained is not None and retained.status is SessionStatus(outcome)
            records = await sessions.query_events(EventQuery(session_id=session.id))
            assert sum(record.event.type is terminal for record in records) == 1
            # Governed task completion remains with the completion-decision owner.
            task = await tasks.load_task(admission.task_id)
            assert task is not None and task.status is TaskStatus.RUNNING
            assert not state.task_finished
        finally:
            _deactivate_session_interaction(session.id)
            _deactivate_session_run_fence(session.id)

    asyncio.run(scenario())
