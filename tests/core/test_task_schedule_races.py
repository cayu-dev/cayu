from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from cayu import CayuApp, TaskCreate, TaskQuery, TaskRescheduleRequest, TaskStatus
from cayu.storage.migrations import SchemaMode
from cayu.storage.postgres import PostgresTaskStore
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks.base import InMemoryTaskStore
from cayu.tasks.scheduling import (
    TaskScheduleCancelRequest,
    TaskScheduleConflict,
    TaskScheduleEventType,
    TaskSchedulePolicy,
    TaskScheduleReceipt,
)


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("competitor", ["cancel", "claim"])
def test_schedule_controllers_and_claims_share_one_revision_authority(
    tmp_path, postgres_dsn, backend, competitor
):
    async def run():
        identity = f"schedule-race-{backend}-{competitor}"
        if backend == "memory":
            first = second = InMemoryTaskStore()
        elif backend == "sqlite":
            first = SQLiteTaskStore(tmp_path / "race.sqlite")
            second = SQLiteTaskStore(tmp_path / "race.sqlite")
        else:
            first = PostgresTaskStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
            second = PostgresTaskStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            apps = [CayuApp(task_store=store, enable_logging=False) for store in (first, second)]
            request = TaskCreate(
                task_id=identity,
                type=identity,
                available_at=datetime.now(UTC),
                schedule_policy=TaskSchedulePolicy(),
            )
            created = await asyncio.gather(*(app.create_task(request) for app in apps))
            assert created[0] == created[1]
            edit = TaskRescheduleRequest(
                task_id=identity,
                operation_id="reschedule",
                expected_revision=1,
                available_at=datetime.now(UTC) + timedelta(hours=1),
            )
            ready = asyncio.Barrier(2)

            async def reschedule():
                await ready.wait()
                return await apps[0].reschedule_task(edit)

            async def compete():
                await ready.wait()
                if competitor == "cancel":
                    return await apps[1].cancel_scheduled_task(
                        TaskScheduleCancelRequest(
                            task_id=identity, operation_id="cancel", expected_revision=1
                        )
                    )
                return await second.claim_task("worker", TaskQuery(type=identity))

            changed, other = await asyncio.gather(reschedule(), compete(), return_exceptions=True)
            task = await first.load_task(identity)
            assert task is not None and task.schedule is not None
            assert task.schedule.revision == 2
            history = await apps[0].list_task_schedule_events(identity)
            assert history[0].type is TaskScheduleEventType.SCHEDULED
            if isinstance(changed, TaskScheduleReceipt):
                assert task.status is TaskStatus.PENDING
                assert task.available_at == edit.available_at
                assert (
                    isinstance(other, TaskScheduleConflict)
                    if competitor == "cancel"
                    else other is None
                )
                assert [event.type for event in history[1:]] == [TaskScheduleEventType.RESCHEDULED]
                assert await apps[1].reschedule_task(edit) == changed
            else:
                assert isinstance(changed, TaskScheduleConflict)
                if competitor == "cancel":
                    assert isinstance(other, TaskScheduleReceipt)
                    assert task.status is TaskStatus.CANCELLED
                    assert [event.type for event in history[1:]] == [
                        TaskScheduleEventType.CANCELLED
                    ]
                else:
                    assert other == task and task.status is TaskStatus.CLAIMED
                    assert [event.type for event in history[1:]] == [
                        TaskScheduleEventType.ELIGIBLE,
                        TaskScheduleEventType.CLAIMED,
                    ]
            assert await second.claim_task("late-worker", TaskQuery(type=identity)) is None
            assert await apps[1].list_task_schedule_events(identity) == history
        finally:
            if isinstance(first, (SQLiteTaskStore, PostgresTaskStore)):
                await first.close()
                await second.close()

    asyncio.run(run())
