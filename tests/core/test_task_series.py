"""Recurring task series: occurrence times, idempotent publication and the handler wrapper."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from cayu.applications import CayuApp
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks import (
    TaskCreate,
    TaskRecurrence,
    TaskSeries,
    last_occurrence,
    next_occurrence,
    occurrence_task_id,
    publish_next_occurrence,
    series_task_handler,
    task_occurrence,
)
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.queries import TaskQuery
from cayu.tasks.records import Task, TaskStatus
from cayu.tasks.scheduling import TaskMisfirePolicy
from cayu.tasks.store import TaskStore
from cayu.tasks.worker import complete_managed_task

START = datetime(2026, 1, 1, tzinfo=UTC)


def _utc(*args: int) -> datetime:
    return datetime(*args, tzinfo=UTC)


def test_cron_occurrences_follow_the_time_zone_and_day_rules() -> None:
    # Mondays at 9:00 in Los Angeles: 17:00 UTC in winter, 16:00 in summer.
    weekly = TaskRecurrence(cron="0 9 * * MON", timezone="America/Los_Angeles", start_at=START)
    assert next_occurrence(weekly, START) == _utc(2026, 1, 5, 17)
    assert next_occurrence(weekly, _utc(2026, 1, 5, 17)) == _utc(2026, 1, 12, 17)
    assert next_occurrence(weekly, _utc(2026, 3, 9)) == _utc(2026, 3, 9, 16)
    assert last_occurrence(weekly, _utc(2026, 1, 12, 17)) == _utc(2026, 1, 12, 17)
    assert last_occurrence(weekly, _utc(2026, 1, 12, 16)) == _utc(2026, 1, 5, 17)
    assert last_occurrence(weekly, _utc(2026, 1, 5, 16)) is None

    monthly = TaskRecurrence(cron="0 6 1 * *", start_at=START)
    assert next_occurrence(monthly, _utc(2026, 1, 31)) == _utc(2026, 2, 1, 6)
    # Both day fields restricted: either matches, as in cron.
    either = TaskRecurrence(cron="0 0 13 * FRI", start_at=START)
    assert next_occurrence(either, START) == _utc(2026, 1, 2)
    steps = TaskRecurrence(cron="*/15 8-9 * JAN,FEB 1-5", start_at=START)
    assert next_occurrence(steps, _utc(2026, 1, 2, 9, 45)) == _utc(2026, 1, 5, 8)

    # 2:30 doesn't exist on the spring-forward day, so that day has no occurrence.
    nightly = TaskRecurrence(cron="30 2 * * *", timezone="America/Los_Angeles", start_at=START)
    assert next_occurrence(nightly, _utc(2026, 3, 8, 0)) == _utc(2026, 3, 9, 9, 30)
    never = TaskRecurrence(cron="0 0 30 FEB *", start_at=START)
    assert next_occurrence(never, START) is None


def test_interval_occurrences_count_from_start() -> None:
    hourly = TaskRecurrence(every_seconds=3600, start_at=START)
    assert next_occurrence(hourly, START - timedelta(days=1)) == START
    assert next_occurrence(hourly, START) == START + timedelta(hours=1)
    assert next_occurrence(hourly, START + timedelta(minutes=90)) == START + timedelta(hours=2)
    assert last_occurrence(hourly, START + timedelta(minutes=90)) == START + timedelta(hours=1)


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"cron": "0 9 * * MON", "every_seconds": 3600},
        {"cron": "0 9 * *"},
        {"cron": "61 * * * *"},
        {"cron": "0 9 * * MON", "timezone": "Mars/Base"},
        {"every_seconds": 1},
    ],
)
def test_invalid_recurrences_are_refused(fields: dict) -> None:
    with pytest.raises(ValidationError):
        TaskRecurrence(start_at=START, **fields)


def test_a_template_leaves_the_occurrence_fields_to_the_series() -> None:
    recurrence = TaskRecurrence(every_seconds=3600, start_at=START)
    for fields in (
        {"task_id": "x"},
        {"available_at": START},
        {"retry_key": "k"},
        {"retry_policy": {"max_attempts": 2}},
    ):
        with pytest.raises(ValidationError):
            TaskSeries(
                series_id="s",
                recurrence=recurrence,
                task=TaskCreate(type="review", **fields),
            )
    with pytest.raises(ValidationError, match="'@'"):
        TaskSeries(series_id="a@b", recurrence=recurrence, task=TaskCreate(type="review"))


def _series(**fields) -> TaskSeries:
    return TaskSeries(
        series_id="review-owner-1",
        recurrence=TaskRecurrence(cron="0 9 * * MON", start_at=START),
        task=TaskCreate(type="review", input={"owner": "1"}, labels={"owner": "1"}),
        **fields,
    )


def _store(backend: str, tmp_path: Path) -> TaskStore:
    if backend == "memory":
        return InMemoryTaskStore()
    return SQLiteTaskStore(tmp_path / "tasks.db")


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_publication_is_idempotent_and_skips_or_catches_up_missed_occurrences(
    backend: str, tmp_path: Path
) -> None:
    async def scenario() -> None:
        store = _store(backend, tmp_path)
        series = _series()
        now = _utc(2025, 12, 30)
        racing = await asyncio.gather(
            *(publish_next_occurrence(store, series, now=now) for _ in range(3))
        )
        assert {task.id for task in racing if task is not None} == {
            "review-owner-1@20260105T090000Z"
        }
        [first] = await store.list_tasks(TaskQuery(type="review"))
        assert first.available_at == _utc(2026, 1, 5, 9)
        assert first.input == {"owner": "1"}
        assert first.invocation.access_labels == {"owner": "1"}
        assert task_occurrence(first) == ("review-owner-1", _utc(2026, 1, 5, 9))

        # Published long after: SKIP goes on with the next future occurrence ...
        late = _utc(2026, 2, 4)
        skipped = await publish_next_occurrence(store, series, after=_utc(2026, 1, 5, 9), now=late)
        assert skipped is not None and skipped.available_at == _utc(2026, 2, 9, 9)
        # ... and FIRE_ONCE queues the latest missed one, once.
        catch_up = _series(misfire_policy=TaskMisfirePolicy.FIRE_ONCE)
        caught = await publish_next_occurrence(store, catch_up, after=_utc(2026, 1, 5, 9), now=late)
        assert caught is not None and caught.id == occurrence_task_id(
            "review-owner-1", _utc(2026, 2, 2, 9)
        )
        close = getattr(store, "close", None)
        if close is not None:
            await close()

    asyncio.run(scenario())


def test_the_handler_wrapper_publishes_the_next_occurrence_first_and_honours_pause() -> None:
    async def scenario() -> None:
        now = datetime.now(UTC).replace(second=0, microsecond=0)
        clock = {"now": now}
        store = InMemoryTaskStore(clock=lambda: clock["now"], ownership_clock=lambda: clock["now"])
        app = CayuApp(task_store=store, enable_logging=False)
        series = TaskSeries(
            series_id="digest",
            recurrence=TaskRecurrence(every_seconds=3600, start_at=now),
            task=TaskCreate(type="digest"),
        )
        paused: set[str] = set()
        ran: list[str] = []

        async def handler(_app: CayuApp, task: Task, worker_id: str) -> None:
            ran.append(task.id)
            await complete_managed_task(store, task, worker_id, {"ok": True})

        wrapped = series_task_handler(
            store, handler, series_for=lambda task: None if paused else series
        )
        first = await publish_next_occurrence(store, series, now=now)
        assert first is not None and first.id == occurrence_task_id("digest", now)
        claimed = await store.claim_task("worker", TaskQuery(type="digest"))
        assert claimed is not None and claimed.id == first.id
        await wrapped(app, claimed, "worker")
        assert ran == [first.id]
        second_id = occurrence_task_id("digest", now + timedelta(hours=1))
        upcoming = await store.load_task(second_id)
        assert upcoming is not None and upcoming.status is TaskStatus.PENDING

        # A paused series completes its queued occurrence without running it,
        # and publishes nothing more.
        paused.add("digest")
        clock["now"] = now + timedelta(hours=1)
        second = await store.claim_task("worker", TaskQuery(type="digest"))
        assert second is not None and second.id == second_id
        await wrapped(app, second, "worker")
        done = await store.load_task(second_id)
        assert done is not None and done.result == {"series_paused": True}
        assert ran == [first.id]
        assert len(await store.list_tasks(TaskQuery(type="digest"))) == 2

        # Ordinary tasks pass straight through.
        await store.create_task(TaskCreate(task_id="plain", type="other"))
        plain = await store.claim_task("worker", TaskQuery(type="other"))
        assert plain is not None
        await wrapped(app, plain, "worker")
        assert ran[-1] == "plain"
        await app.aclose()

    asyncio.run(scenario())


def test_a_late_occurrence_of_a_skip_series_is_skipped_but_the_series_goes_on() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        start = datetime.now(UTC).replace(microsecond=0) - timedelta(hours=2, minutes=30)
        series = TaskSeries(
            series_id="late",
            recurrence=TaskRecurrence(every_seconds=3600, start_at=start),
            task=TaskCreate(type="late"),
        )
        ran: list[str] = []

        async def handler(_app: CayuApp, task: Task, worker_id: str) -> None:
            ran.append(task.id)

        # Published on time, but no worker ran it until two and a half hours later.
        await publish_next_occurrence(store, series, now=start)
        claimed = await store.claim_task("worker", TaskQuery(type="late"))
        assert claimed is not None
        await series_task_handler(store, handler, series_for=lambda task: series)(
            app, claimed, "worker"
        )
        assert ran == []
        skipped = await store.load_task(claimed.id)
        assert skipped is not None and skipped.result == {"series_skipped": True}
        pending = await store.list_tasks(TaskQuery(type="late", status=TaskStatus.PENDING))
        assert [task.available_at for task in pending] == [start + timedelta(hours=3)]
        await app.aclose()

    asyncio.run(scenario())


def test_resuming_steps_over_a_cancelled_future_occurrence() -> None:
    async def scenario() -> None:
        from cayu.tasks.scheduling import TaskScheduleCancelRequest

        store = InMemoryTaskStore()
        series = _series()
        now = _utc(2025, 12, 30)
        queued = await publish_next_occurrence(store, series, now=now)
        assert queued is not None and queued.schedule is not None
        await store.cancel_scheduled_task(
            TaskScheduleCancelRequest(
                task_id=queued.id, operation_id="pause", expected_revision=queued.schedule.revision
            )
        )
        resumed = await publish_next_occurrence(store, series, after=now, now=now)
        assert resumed is not None and resumed.available_at == _utc(2026, 1, 12, 9)
        assert resumed.status is TaskStatus.PENDING

    asyncio.run(scenario())
