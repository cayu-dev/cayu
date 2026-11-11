"""Managed schedule admission is consistent across task-store backends."""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from tests.core.task_invocation_fixtures import task_backed_session_invocation
from tests.core.test_verified_work_contracts import _contract
from tests.core.test_work_attempt_admission import _prepare_request
from tests.core.verified_worker_fixtures import (
    verified_work_postgres_dsn as verified_work_postgres_dsn,
)

from cayu.storage.migrations import SchemaMode
from cayu.storage.tasks_postgres import PostgresTaskStore
from cayu.storage.tasks_sqlite import SQLiteTaskStore
from cayu.tasks.creation import TaskCreate
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.scheduling import (
    TaskMisfirePolicy,
    TaskScheduleConflict,
    TaskScheduleEventType,
    TaskSchedulePolicy,
)


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("case", ["expired", "skipped", "eligible", "fire_once", "claimed"])
def test_verified_schedule_admission_backend_parity(backend, case, tmp_path, request):
    dsn = request.getfixturevalue("verified_work_postgres_dsn") if backend == "postgres" else None

    async def scenario():
        clock = [datetime(2026, 1, 1, tzinfo=UTC)]
        if backend == "memory":
            store = InMemoryTaskStore(clock=lambda: clock[0], ownership_clock=lambda: clock[0])
        elif backend == "sqlite":
            store = SQLiteTaskStore(
                tmp_path / "schedule.sqlite",
                clock=lambda: clock[0],
                ownership_clock=lambda: clock[0],
            )
        else:
            store = PostgresTaskStore(dsn, schema_mode=SchemaMode.CREATE, clock=lambda: clock[0])
        try:
            contract = await store.publish_work_contract(_contract())
            task = await store.create_task(
                TaskCreate(
                    task_id="scheduled-work",
                    type="verified-work",
                    work_contract=contract.reference(),
                    available_at=clock[0]
                    if case in {"eligible", "claimed"}
                    else clock[0] - timedelta(minutes=2),
                    schedule_policy=TaskSchedulePolicy(
                        expires_at=clock[0] - timedelta(minutes=1)
                        if case == "expired"
                        else clock[0] + timedelta(seconds=1),
                        misfire_policy=TaskMisfirePolicy.SKIP
                        if case == "skipped"
                        else TaskMisfirePolicy.FIRE_ONCE,
                        misfire_grace_seconds=30,
                    ),
                )
            )
            if case == "claimed":
                task = await store.claim_task("worker-1", lease_seconds=300)
                clock[0] += timedelta(seconds=2)
            prepare = _prepare_request(
                task_id=task.id,
                session_id="scheduled-session",
                session_invocation=await task_backed_session_invocation(
                    store, task.id, "scheduled-session"
                ),
                task_lease_expires_at=task.lease_expires_at,
            )
            history = await store.list_task_schedule_events(task.id)
            if case in {"expired", "skipped"}:
                with pytest.raises(TaskScheduleConflict):
                    await store.prepare_work_attempt_admission(prepare)
                assert await store.load_work_attempt_admission(prepare.admission_id) is None
                assert await store.load_task(task.id) == task
                assert await store.list_task_schedule_events(task.id) == history
            else:
                admission = await store.prepare_work_attempt_admission(prepare)
                admitted = await store.load_task(task.id)
                if case == "claimed":
                    assert admitted.schedule == task.schedule
                else:
                    assert admitted.schedule.admitted_at == clock[0]
                    assert admitted.schedule.revision == task.schedule.revision + 1
                history = await store.list_task_schedule_events(task.id)
                assert [event.type for event in history].count(
                    TaskScheduleEventType.MISFIRED
                    if case == "fire_once"
                    else TaskScheduleEventType.ELIGIBLE
                ) == 1
                clock[0] += timedelta(minutes=1)
                assert await store.prepare_work_attempt_admission(prepare) == admission
                assert await store.load_task(task.id) == admitted
                assert await store.list_task_schedule_events(task.id) == history
        finally:
            if backend != "memory":
                await store.close()

    asyncio.run(scenario())
