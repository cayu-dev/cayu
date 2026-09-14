from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from cayu import AgentSpec, CayuApp, Message, RunRequest, TaskCreate, TaskQuery, TaskStatus
from cayu.evals.testing import ScriptedModelProvider
from cayu.events import EventType
from cayu.providers.base import ModelStreamEvent
from cayu.sessions.base import InMemorySessionStore
from cayu.storage.migrations import SchemaMode
from cayu.storage.postgres import PostgresSessionStore, PostgresTaskStore
from cayu.storage.sqlite import SQLiteSessionStore, SQLiteTaskStore
from cayu.tasks.base import InMemoryTaskStore
from cayu.tasks.scheduling import TaskScheduleEventType, TaskSchedulePolicy
from cayu.tasks.worker import run_task_worker


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("provider_fails", [False, True])
def test_scheduled_worker_run_preserves_one_execution_and_terminal_evidence(
    tmp_path, postgres_dsn, backend, provider_fails
):
    async def run():
        identity = f"scheduled-runtime-{backend}-{provider_fails}"
        if backend == "memory":
            tasks = InMemoryTaskStore()
            sessions = InMemorySessionStore()
        elif backend == "sqlite":
            tasks = SQLiteTaskStore(tmp_path / "runtime.sqlite")
            sessions = SQLiteSessionStore(tmp_path / "runtime.sqlite")
        else:
            tasks = PostgresTaskStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
            sessions = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            app = CayuApp(task_store=tasks, session_store=sessions, enable_logging=False)
            provider = ScriptedModelProvider(
                [[ModelStreamEvent.error("provider unavailable"), ModelStreamEvent.completed()]]
                if provider_fails
                else [[ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()]]
            )
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="worker", model="scripted-model"))
            await app.create_task(
                TaskCreate(
                    task_id=identity,
                    type=identity,
                    available_at=datetime.now(UTC),
                    schedule_policy=TaskSchedulePolicy(),
                )
            )

            async def handler(app, task, worker_id):
                async for _ in app.run(
                    RunRequest(
                        agent_name="worker",
                        session_id=f"session-{identity}",
                        task_id=task.id,
                        task_worker_id=worker_id,
                        task_lease_expires_at=task.lease_expires_at,
                        messages=[Message.text("user", "perform the follow-up")],
                    )
                ):
                    pass

            assert (
                await run_task_worker(
                    app,
                    tasks,
                    handler,
                    worker_id="worker",
                    query=TaskQuery(type=identity),
                    max_tasks=1,
                    reclaim=False,
                )
                == 1
            )
            terminal = await tasks.load_task(identity)
            assert terminal is not None
            assert terminal.status is (
                TaskStatus.FAILED if provider_fails else TaskStatus.COMPLETED
            )
            assert len(provider.requests) == 1
            history = await app.list_task_schedule_events(identity)
            assert [event.type for event in history] == [
                TaskScheduleEventType.SCHEDULED,
                TaskScheduleEventType.ELIGIBLE,
                TaskScheduleEventType.CLAIMED,
                TaskScheduleEventType.STARTED,
                TaskScheduleEventType.FAILED if provider_fails else TaskScheduleEventType.COMPLETED,
            ]
            events = await sessions.load_events(f"session-{identity}")
            expected = EventType.SESSION_FAILED if provider_fails else EventType.SESSION_COMPLETED
            assert sum(event.type is expected for event in events) == 1
            assert await tasks.claim_task("replacement", TaskQuery(type=identity)) is None
        finally:
            if isinstance(tasks, (SQLiteTaskStore, PostgresTaskStore)):
                await tasks.close()
                await sessions.close()

    asyncio.run(run())
