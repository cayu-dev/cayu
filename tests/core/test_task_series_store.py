"""Task series kept by the task store: each ended occurrence queues the next in its transaction."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from cayu.applications import CayuApp
from cayu.storage.migrations import SchemaMode
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks import (
    TaskCreate,
    TaskRecurrence,
    TaskSeries,
    occurrence_task_id,
    series_task_handler,
)
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.queries import TaskQuery
from cayu.tasks.records import Task, TaskStatus
from cayu.tasks.scheduling import TaskScheduleCancelRequest
from cayu.tasks.store import TaskStore

BACKENDS = ["memory", "sqlite", "postgres"]


def _store(backend: str, tmp_path: Path, request: pytest.FixtureRequest) -> TaskStore:
    if backend == "memory":
        return InMemoryTaskStore()
    if backend == "sqlite":
        return SQLiteTaskStore(tmp_path / "tasks.db")
    from cayu.storage import PostgresTaskStore

    dsn = request.getfixturevalue("postgres_dsn")
    return PostgresTaskStore(dsn, schema_mode=SchemaMode.MIGRATE)


async def _close(store: TaskStore) -> None:
    close = getattr(store, "close", None)
    if close is not None:
        await close()


def _series(series_id: str, start: datetime, **fields) -> TaskSeries:
    return TaskSeries(
        series_id=series_id,
        recurrence=TaskRecurrence(every_seconds=3600, start_at=start),
        task=TaskCreate(type=f"digest-{series_id}", labels={"owner": "1"}),
        **fields,
    )


async def _finish(store: TaskStore, task_type: str, *, fail: bool = False) -> Task:
    claimed = await store.claim_task("worker", TaskQuery(type=task_type))
    assert claimed is not None
    if fail:
        return await store.fail_task(
            claimed.id,
            {"message": "failed"},
            worker_id="worker",
            claim_generation=claimed.claim_generation,
        )
    return await store.complete_task(
        claimed.id, {"ok": True}, worker_id="worker", claim_generation=claimed.claim_generation
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_each_ended_occurrence_queues_the_next_until_paused_or_deleted(
    backend: str, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    async def scenario() -> None:
        store = _store(backend, tmp_path, request)
        start = datetime.now(UTC).replace(microsecond=0) - timedelta(seconds=5)
        series = _series("s1", start)
        kind = "digest-s1"
        assert await store.load_task_series("s1") is None

        saved = await store.save_task_series(series)
        first_id = occurrence_task_id("s1", start)
        assert (saved.revision, saved.paused, saved.next_task_id) == (1, False, first_id)
        assert await store.load_task_series("s1") == saved
        # Saving again keeps the occurrence already queued.
        again = await store.save_task_series(series)
        assert again.revision == 2 and again.next_task_id == first_id

        done = await _finish(store, kind)
        assert done.id == first_id
        second_id = occurrence_task_id("s1", start + timedelta(hours=1))
        state = await store.load_task_series("s1")
        assert state is not None and state.next_task_id == second_id
        queued = await store.load_task(second_id)
        assert queued is not None and queued.status is TaskStatus.PENDING
        assert queued.available_at == start + timedelta(hours=1)
        assert queued.invocation.access_labels == {"owner": "1"}

        # A paused series queues nothing more; resuming queues the next after now.
        paused = await store.set_task_series_paused("s1", True)
        assert paused.paused and paused.next_task_id == second_id
        resumed = await store.set_task_series_paused("s1", False)
        assert not resumed.paused and resumed.next_task_id == second_id
        with pytest.raises(KeyError):
            await store.set_task_series_paused("missing", True)

        assert await store.delete_task_series("s1") is True
        assert await store.delete_task_series("s1") is False
        assert await store.load_task_series("s1") is None
        assert len(await store.list_tasks(TaskQuery(type=kind))) == 2
        await _close(store)

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", BACKENDS)
def test_failure_queues_the_next_and_pause_stops_it(
    backend: str, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    async def scenario() -> None:
        store = _store(backend, tmp_path, request)
        start = datetime.now(UTC).replace(microsecond=0) - timedelta(seconds=5)
        await store.save_task_series(_series("s2", start))
        await _finish(store, "digest-s2", fail=True)
        state = await store.load_task_series("s2")
        assert state is not None
        assert state.next_task_id == occurrence_task_id("s2", start + timedelta(hours=1))

        # Paused: ending the queued occurrence (here cancelled) queues nothing.
        await store.set_task_series_paused("s2", True)
        queued = await store.load_task(state.next_task_id)
        assert queued is not None and queued.schedule is not None
        await store.cancel_scheduled_task(
            TaskScheduleCancelRequest(
                task_id=queued.id, operation_id="pause", expected_revision=queued.schedule.revision
            )
        )
        paused = await store.load_task_series("s2")
        assert paused is not None and paused.next_task_id == state.next_task_id

        # Resuming steps over the cancelled future occurrence.
        resumed = await store.set_task_series_paused("s2", False)
        assert resumed.next_task_id == occurrence_task_id("s2", start + timedelta(hours=2))
        tasks = await store.list_tasks(TaskQuery(type="digest-s2"))
        assert sorted(task.status.value for task in tasks) == ["cancelled", "failed", "pending"]
        await _close(store)

    asyncio.run(scenario())


def test_an_older_occurrence_ending_late_does_not_fork_the_series() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        start = datetime.now(UTC).replace(microsecond=0) - timedelta(seconds=5)
        # An occurrence queued by hand for an earlier time, before the series was stored.
        earlier = (start - timedelta(hours=1)).isoformat()
        await store.create_task(
            TaskCreate(
                task_id="by-hand",
                type="stale",
                metadata={"cayu_series": {"series_id": "s3", "occurrence_at": earlier}},
            )
        )
        saved = await store.save_task_series(_series("s3", start))
        await _finish(store, "stale")
        after = await store.load_task_series("s3")
        assert after is not None and after.next_task_id == saved.next_task_id
        assert len(await store.list_tasks(TaskQuery(type="digest-s3"))) == 1

    asyncio.run(scenario())


def test_the_wrapper_reads_the_stored_series() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        start = datetime.now(UTC).replace(microsecond=0) - timedelta(seconds=5)
        await store.save_task_series(_series("s4", start))
        ran: list[str] = []

        async def handler(_app: CayuApp, task: Task, worker_id: str) -> None:
            ran.append(task.id)
            await store.complete_task(
                task.id, {}, worker_id=worker_id, claim_generation=task.claim_generation
            )

        wrapped = series_task_handler(store, handler)
        await store.set_task_series_paused("s4", True)
        claimed = await store.claim_task("worker", TaskQuery(type="digest-s4"))
        assert claimed is not None
        await wrapped(app, claimed, "worker")
        assert ran == []
        done = await store.load_task(claimed.id)
        assert done is not None and done.result == {"series_paused": True}
        # Paused, so the store queued nothing after it.
        assert len(await store.list_tasks(TaskQuery(type="digest-s4"))) == 1
        await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", BACKENDS)
def test_saving_again_steps_over_an_occurrence_that_already_ended(
    backend: str, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    async def scenario() -> None:
        store = _store(backend, tmp_path, request)
        start = datetime.now(UTC).replace(microsecond=0) - timedelta(seconds=5)
        series = _series("s5", start, misfire_policy="fire_once", misfire_grace_seconds=86_400)
        await store.save_task_series(series)
        # Deleted while its occurrence ran: that occurrence ends without a successor.
        assert await store.delete_task_series("s5") is True
        await _finish(store, "digest-s5")
        assert len(await store.list_tasks(TaskQuery(type="digest-s5"))) == 1
        # Stored again, the series steps over the ended occurrence instead of adopting it.
        saved = await store.save_task_series(series)
        assert saved.next_task_id == occurrence_task_id("s5", start + timedelta(hours=1))
        queued = await store.load_task(saved.next_task_id)
        assert queued is not None and queued.status is TaskStatus.PENDING
        await _close(store)

    asyncio.run(scenario())


def test_a_template_an_occurrence_could_not_use_is_refused() -> None:
    try:
        TaskCreate(
            task_id="probe",
            type="probe",
            available_at=datetime.now(UTC),
            schedule_policy={},
            restartable=True,
        )
    except ValueError:
        restartable_schedules = False
    else:
        restartable_schedules = True
    if restartable_schedules:
        pytest.skip("This build accepts restartable scheduled tasks.")
    with pytest.raises(ValueError, match="Restartable"):
        TaskSeries(
            series_id="s6",
            recurrence=TaskRecurrence(every_seconds=3600, start_at=datetime.now(UTC)),
            task=TaskCreate(type="digest", restartable=True),
        )
