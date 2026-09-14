from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from cayu import CayuApp, TaskCreate, TaskQuery, TaskRescheduleRequest, TaskStatus
from cayu.storage.migrations import SchemaMode
from cayu.storage.postgres import PostgresTaskStore
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks.base import InMemoryTaskStore
from cayu.tasks.scheduling import TaskScheduleEventType, TaskSchedulePolicy
from cayu.tasks.worker import complete_managed_task, run_task_worker


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("without_notifications", [False, True])
@pytest.mark.parametrize("observation_delay", [0, 0.4, 0.8])
def test_worker_reobserves_earlier_schedule_with_or_without_wakeup_hints(
    tmp_path, postgres_dsn, backend, without_notifications, observation_delay, monkeypatch
):
    async def run():
        identity = f"earlier-schedule-{backend}-{without_notifications}-{observation_delay}"
        if backend == "memory":
            store = InMemoryTaskStore()
        elif backend == "sqlite":
            store = SQLiteTaskStore(tmp_path / "wakeup.sqlite")
        else:
            store = PostgresTaskStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        worker = None
        try:
            app = CayuApp(task_store=store, enable_logging=False)
            original_due = datetime.now(UTC) + timedelta(minutes=1)
            await app.create_task(
                TaskCreate(
                    task_id=identity,
                    type=identity,
                    available_at=original_due,
                    schedule_policy=TaskSchedulePolicy(),
                )
            )
            observed_original = asyncio.Event()
            original_observe = store.next_task_schedule_wakeup

            async def observe(query=None):
                # A database may obtain its authoritative clock only after
                # waiting for a connection or preceding commands to finish.
                if observation_delay:
                    await asyncio.sleep(observation_delay)
                result = await original_observe(query)
                if result.next_available_at == original_due:
                    observed_original.set()
                return result

            monkeypatch.setattr(store, "next_task_schedule_wakeup", observe)
            if without_notifications:
                # Real worker fallback entrance: no subscription is available.
                # Store deadlines are still observed, but no hint wakes the loop.
                async def unavailable_subscription(queries):
                    return None

                monkeypatch.setattr(store, "_task_admission_wakeup", unavailable_subscription)
            calls = []

            async def handler(_app, task, worker_id):
                calls.append(datetime.now(UTC))
                await complete_managed_task(store, task, worker_id, {"done": True})

            poll = 0.25 if without_notifications else 10.0
            worker = asyncio.create_task(
                run_task_worker(
                    app,
                    store,
                    handler,
                    worker_id="worker",
                    query=TaskQuery(type=identity),
                    poll_interval_s=poll,
                    minimum_idle_delay_s=poll,
                    maximum_idle_delay_s=poll,
                    idle_jitter_ratio=0,
                    reclaim=False,
                    recover_interrupted_handoffs=False,
                    max_tasks=1,
                )
            )
            await asyncio.wait_for(observed_original.wait(), timeout=5)
            assert not calls
            replacement_due = datetime.now(UTC) + timedelta(seconds=0.6)
            receipt = await app.reschedule_task(
                TaskRescheduleRequest(
                    task_id=identity,
                    operation_id="move-earlier",
                    expected_revision=1,
                    available_at=replacement_due,
                )
            )
            assert receipt.schedule.revision == 2
            # Eligibility may cross during the read, leaving no future deadline
            # to return. Notifications are advisory: allow the configured fallback
            # poll plus store/handler settlement, not less than the poll interval.
            assert await asyncio.wait_for(asyncio.shield(worker), timeout=poll + 5) == 1
            assert len(calls) == 1 and calls[0] >= replacement_due
            task = await store.load_task(identity)
            assert task is not None and task.status is TaskStatus.COMPLETED
            history = await app.list_task_schedule_events(identity)
            assert [event.type for event in history] == [
                TaskScheduleEventType.SCHEDULED,
                TaskScheduleEventType.RESCHEDULED,
                TaskScheduleEventType.ELIGIBLE,
                TaskScheduleEventType.CLAIMED,
                TaskScheduleEventType.STARTED,
                TaskScheduleEventType.COMPLETED,
            ]
        finally:
            if worker is not None:
                if not worker.done():
                    worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
            if isinstance(store, (SQLiteTaskStore, PostgresTaskStore)):
                await store.close()

    asyncio.run(run())


def test_schedule_deadline_anchors_to_observation_response(monkeypatch):
    from cayu.tasks import _schedule_wakeup
    from cayu.tasks.scheduling import TaskScheduleWakeup

    ticks = [100.0]
    as_of = datetime(2026, 9, 14, tzinfo=UTC)

    class DelayedObservationStore(InMemoryTaskStore):
        async def next_task_schedule_wakeup(self, query=None):
            await asyncio.sleep(0)
            ticks[0] = 101.0
            return TaskScheduleWakeup(as_of=as_of, next_available_at=as_of + timedelta(seconds=2))

    monkeypatch.setattr(_schedule_wakeup, "monotonic", lambda: ticks[0])
    deadline = asyncio.run(
        _schedule_wakeup.next_schedule_wake_at(DelayedObservationStore(), (None,))
    )
    # Request-start anchoring would return 102 and wake before the store's due
    # time. Pin this arithmetic independently of the integration polling fallback.
    assert deadline == 103.0
