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
