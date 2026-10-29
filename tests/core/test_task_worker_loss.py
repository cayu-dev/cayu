"""Tasks fenced after their worker was lost: filter, settle and warn, on real stores."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from cayu.applications import CayuApp
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks import (
    is_fenced_after_worker_loss,
    reconcile_abandoned_tasks,
)
from cayu.tasks.abandoned import (
    RECONCILE_ABANDONED_AFTER_ENV,
    SINGLE_WORKER_VALIDATOR_ID,
)
from cayu.tasks.creation import TaskCreate
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.queries import TaskOrder, TaskQuery
from cayu.tasks.records import Task, TaskStatus
from cayu.tasks.store import TaskStore
from cayu.tasks.worker import run_task_worker

LEASE_SECONDS = 30


@pytest.fixture(autouse=True)
def _no_ambient_declaration(monkeypatch: pytest.MonkeyPatch) -> None:
    # Tests here expect no single-worker declaration unless they set one.
    monkeypatch.delenv(RECONCILE_ABANDONED_AFTER_ENV, raising=False)


class _Clock:
    """The store's ownership clock, an hour behind real time so wall-clock checks agree."""

    def __init__(self) -> None:
        self.value = datetime.now(UTC).replace(microsecond=0) - timedelta(hours=1)

    def __call__(self) -> datetime:
        return self.value


def _store(backend: str, tmp_path: Path, clock: _Clock) -> TaskStore:
    if backend == "memory":
        return InMemoryTaskStore(ownership_clock=clock)
    return SQLiteTaskStore(tmp_path / "tasks.db", ownership_clock=clock)


async def _started_claim(store: TaskStore, task_id: str, worker_id: str) -> Task:
    """Create a task and claim it, then mark its handler dispatched, as a worker does."""

    await store.create_task(TaskCreate(task_id=task_id, type="scan"))
    claimed = await store.claim_task(worker_id, TaskQuery(type="scan"), lease_seconds=LEASE_SECONDS)
    assert claimed is not None and claimed.id == task_id
    assert claimed.lease_expires_at is not None
    return await store.mark_claimed_task_execution_started(
        task_id, worker_id, claimed.lease_expires_at
    )


async def _fence_after_worker_loss(store: TaskStore, clock: _Clock, started: list[Task]) -> None:
    """Let the started claims' leases expire and reclaim, as the next worker would."""

    assert all(task.lease_expires_at is not None for task in started)
    clock.value = max(task.lease_expires_at for task in started) + timedelta(seconds=1)  # type: ignore[type-var]
    assert await store.reclaim_expired(query=TaskQuery(type="scan")) == []


def _ids(tasks: list[Task]) -> list[str]:
    return sorted(task.id for task in tasks)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_fenced_tasks_are_told_apart_from_live_claims(backend: str, tmp_path: Path) -> None:
    async def scenario() -> None:
        clock = _Clock()
        store = _store(backend, tmp_path, clock)
        dead = await _started_claim(store, "dead", "worker-a")
        await _fence_after_worker_loss(store, clock, [dead])
        fenced = await store.load_task("dead")
        assert fenced is not None
        assert fenced.status is TaskStatus.CLAIMED
        assert fenced.status_reason == "cancellation_requested"
        assert fenced.status_payload is not None
        assert fenced.status_payload["error"] == {
            "code": "task_worker_lease_expired_after_dispatch"
        }

        live = await _started_claim(store, "live", "worker-b")
        # A live worker draining a cancellation keeps its lease: not fenced yet.
        draining = await _started_claim(store, "draining", "worker-c")
        draining = await store.cancel_task("draining", {"code": "worker_cancelled"})
        assert draining.status_reason == "cancellation_requested"
        await store.create_task(TaskCreate(task_id="queued", type="scan"))

        now = clock.value
        assert is_fenced_after_worker_loss(fenced, now)
        assert not is_fenced_after_worker_loss(live, now)
        assert not is_fenced_after_worker_loss(draining, now)
        assert draining.lease_expires_at is not None
        assert is_fenced_after_worker_loss(draining, draining.lease_expires_at)

        assert _ids(await store.list_tasks(TaskQuery(fenced_after_worker_loss=True))) == ["dead"]
        assert _ids(await store.list_tasks(TaskQuery(fenced_after_worker_loss=False))) == [
            "draining",
            "live",
            "queued",
        ]
        # "Is a run already queued or running?" no longer counts the fenced task.
        assert _ids(
            await store.list_tasks(
                TaskQuery(status=TaskStatus.CLAIMED, fenced_after_worker_loss=False)
            )
        ) == ["draining", "live"]

        # Once the draining worker's lease expires too, it is fenced by the store's clock.
        clock.value = draining.lease_expires_at
        assert _ids(await store.list_tasks(TaskQuery(fenced_after_worker_loss=True))) == [
            "dead",
            "draining",
        ]
        with pytest.raises(ValueError, match="fenced_after_worker_loss"):
            await store.claim_task("worker-d", TaskQuery(fenced_after_worker_loss=True))
        close = getattr(store, "close", None)
        if close is not None:
            await close()

    asyncio.run(scenario())


def test_reconcile_abandoned_tasks_settles_fences_of_gone_workers_only(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        clock = _Clock()
        store = SQLiteTaskStore(tmp_path / "tasks.db", ownership_clock=clock)
        gone = await _started_claim(store, "gone", "worker-old")
        held_live = await _started_claim(store, "held-live", "worker-live")
        await _fence_after_worker_loss(store, clock, [gone, held_live])
        # A cancellation another worker is still draining under a live lease.
        await _started_claim(store, "draining", "worker-other")
        await store.cancel_task("draining", {"code": "worker_cancelled"})
        assert gone.lease_expires_at is not None
        settle = {
            "live_worker_ids": ["worker-live"],
            "validator_id": "service.single-worker",
            "validator_version": "2",
        }

        [settled] = await reconcile_abandoned_tasks(store, **settle)
        assert settled.id == "gone"
        assert settled.status is TaskStatus.CANCELLED
        assert settled.worker_id is None
        assert settled.status_payload is not None
        evidence = settled.status_payload["cancellation_reconciliation"]["evidence"]
        assert evidence["validator_id"] == "service.single-worker"
        assert evidence["validator_version"] == "2"
        assert evidence["evidence_id"] == (f"lease-expired-at:{gone.lease_expires_at.isoformat()}")
        # The live worker's task stays fenced, the draining one keeps its lease, and a
        # second pass settles nothing new.
        for task_id in ("held-live", "draining"):
            kept = await store.load_task(task_id)
            assert kept is not None and kept.status is TaskStatus.CLAIMED
        assert await reconcile_abandoned_tasks(store, **settle) == []

        with pytest.raises(TypeError, match="live_worker_ids"):
            await reconcile_abandoned_tasks(store, **{**settle, "live_worker_ids": "worker-live"})
        with pytest.raises(ValueError, match="only considers fenced"):
            await reconcile_abandoned_tasks(
                store, query=TaskQuery(fenced_after_worker_loss=False), **settle
            )
        await store.close()

    asyncio.run(scenario())


def test_the_declared_wait_counts_from_the_first_scan_whatever_the_store_clock(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import cayu.tasks.abandoned as abandoned

    caplog.set_level(logging.WARNING, logger="cayu.tasks.abandoned")

    async def scenario() -> None:
        clock = _Clock()
        # The store's clock runs five minutes ahead of the worker's.
        clock.value += timedelta(minutes=5)
        store = InMemoryTaskStore(ownership_clock=clock)
        dead = await _started_claim(store, "dead", "worker-old")
        await _fence_after_worker_loss(store, clock, [dead])
        elapsed = [0.0]
        monitor = abandoned.FencedTaskMonitor(
            store,
            worker_id="worker-new",
            query=TaskQuery(type="scan"),
            reconcile_abandoned_after_s=120,
            clock=lambda: elapsed[0],
        )
        # Fenced one second ago by the store's clock: the wait starts now.
        assert await monitor.scan() == []
        elapsed[0] = 119.0
        assert await monitor.scan() == []
        elapsed[0] = 120.0
        [settled] = await monitor.scan()
        assert settled.id == "dead" and settled.status is TaskStatus.CANCELLED

    asyncio.run(scenario())
    [message] = _warnings(caplog, "is fenced")
    assert "once it has stayed fenced for 120 seconds" in message


def test_counts_and_listings_keep_the_store_expiry_decision(tmp_path: Path) -> None:
    from cayu.cloud_reporting import read_cloud_task_health

    async def scenario() -> None:
        clock = _Clock()
        # Ahead of local time: the lease expired by the store's clock, not by ours.
        clock.value = datetime.now(UTC) + timedelta(minutes=5)
        store = InMemoryTaskStore(ownership_clock=clock)
        dead = await _started_claim(store, "dead", "worker-old")
        await _fence_after_worker_loss(store, clock, [dead])
        [listed] = await store.list_tasks(TaskQuery(fenced_after_worker_loss=True))
        assert not is_fenced_after_worker_loss(listed)
        health = await read_cloud_task_health(CayuApp(task_store=store, enable_logging=False))
        assert health is not None and health.fenced_after_worker_loss == 1

    asyncio.run(scenario())


async def _run_worker_until(
    app: CayuApp,
    store: TaskStore,
    done: Callable[[], bool],
    **options: object,
) -> None:
    stop = asyncio.Event()

    async def handler(_app: CayuApp, task: Task, _worker_id: str) -> None:
        raise AssertionError(f"No task is claimable, but {task.id} was handed out.")

    async def watch() -> None:
        while not done():
            await asyncio.sleep(0.01)
        stop.set()

    watcher = asyncio.create_task(watch())
    try:
        await asyncio.wait_for(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="worker-new",
                query=TaskQuery(type="scan"),
                poll_interval_s=0.02,
                reclaim_every_s=0.02,
                recover_interrupted_handoffs=False,
                stop=stop,
                **options,  # type: ignore[arg-type]
            ),
            timeout=10,
        )
    finally:
        watcher.cancel()


def _warnings(caplog: pytest.LogCaptureFixture, text: str) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING and text in record.getMessage()
    ]


def test_next_worker_warns_once_about_a_fenced_task(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="cayu.tasks.abandoned")

    async def scenario() -> None:
        clock = _Clock()
        store = SQLiteTaskStore(tmp_path / "tasks.db", ownership_clock=clock)
        app = CayuApp(task_store=store, enable_logging=False)
        dead = await _started_claim(store, "dead", "worker-old")
        await _fence_after_worker_loss(store, clock, [dead])
        scans = 0
        original_list = store.list_tasks

        async def counting_list(query: TaskQuery | None = None, **kwargs):
            nonlocal scans
            if query is not None and query.fenced_after_worker_loss:
                scans += 1
            return await original_list(query, **kwargs)

        store.list_tasks = counting_list  # type: ignore[method-assign]
        try:
            await _run_worker_until(app, store, lambda: scans >= 3)
        finally:
            await app.aclose()
        loaded = await original_list(TaskQuery(order_by=TaskOrder.CREATED_AT_ASC))
        assert [(task.id, task.status) for task in loaded] == [("dead", TaskStatus.CLAIMED)]
        await store.close()

    asyncio.run(scenario())
    [message] = _warnings(caplog, "is fenced")
    assert "Task dead (type scan)" in message
    assert "worker worker-old" in message
    assert "reconcile_abandoned_task_cancellation" in message
    assert "--reconcile-abandoned-after-seconds" in message


@pytest.mark.parametrize("configured_by", ["argument", "environment"])
def test_declared_single_worker_settles_fenced_tasks_of_other_workers(
    configured_by: str,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caplog.set_level(logging.WARNING, logger="cayu.tasks.abandoned")
    options: dict[str, object] = {}
    if configured_by == "argument":
        options["reconcile_abandoned_after_s"] = 0
    else:
        monkeypatch.setenv(RECONCILE_ABANDONED_AFTER_ENV, "0")

    async def scenario() -> None:
        clock = _Clock()
        store = SQLiteTaskStore(tmp_path / "tasks.db", ownership_clock=clock)
        app = CayuApp(task_store=store, enable_logging=False)
        dead = await _started_claim(store, "dead", "worker-old")
        own = await _started_claim(store, "own", "worker-new")
        await _fence_after_worker_loss(store, clock, [dead, own])
        assert dead.lease_expires_at is not None
        settled: list[Task] = []

        async def cancelled() -> bool:
            task = await store.load_task("dead")
            assert task is not None
            if task.status is TaskStatus.CANCELLED:
                settled.append(task)
            return bool(settled)

        state = {"done": False}

        async def poll() -> None:
            while not await cancelled():
                await asyncio.sleep(0.01)
            # One more scan so the remaining fence is reported.
            await asyncio.sleep(0.1)
            state["done"] = True

        poller = asyncio.create_task(poll())
        try:
            await _run_worker_until(app, store, lambda: state["done"], **options)
        finally:
            poller.cancel()
            await app.aclose()
        [task] = settled
        evidence = task.status_payload["cancellation_reconciliation"]["evidence"]  # type: ignore[index]
        assert evidence["validator_id"] == SINGLE_WORKER_VALIDATOR_ID
        # A fence under the worker's own id is never settled by the declaration.
        kept = await store.load_task("own")
        assert kept is not None and kept.status is TaskStatus.CLAIMED
        await store.close()

    asyncio.run(scenario())
    [settled_message] = _warnings(caplog, "Settled task dead")
    assert "worker worker-new is declared the only worker" in settled_message
    [own_message] = _warnings(caplog, "Task own")
    assert "this worker's own id" in own_message


def test_worker_rejects_an_invalid_declaration(monkeypatch: pytest.MonkeyPatch) -> None:
    store = InMemoryTaskStore()
    app = CayuApp(task_store=store, enable_logging=False)

    async def handler(_app: CayuApp, _task: Task, _worker_id: str) -> None:
        return None

    async def run(**options: object) -> int:
        return await run_task_worker(
            app,
            store,
            handler,
            worker_id="worker",
            max_tasks=0,
            **options,  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="reconcile_abandoned_after_s"):
        asyncio.run(run(reconcile_abandoned_after_s=-1))
    with pytest.raises(ValueError, match="reconcile_abandoned_after_s"):
        asyncio.run(run(reconcile_abandoned_after_s=float("inf")))
    monkeypatch.setenv(RECONCILE_ABANDONED_AFTER_ENV, "soon")
    with pytest.raises(ValueError, match=RECONCILE_ABANDONED_AFTER_ENV):
        asyncio.run(run())


def test_postgres_lists_fenced_tasks_by_its_own_clock(postgres_dsn: str) -> None:
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.tasks_postgres import PostgresTaskStore

    async def scenario() -> None:
        store = PostgresTaskStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            await store._ensure_ready()
            async with store._connection() as conn, conn.cursor() as cur:
                await cur.execute("TRUNCATE cayu_tasks CASCADE")
                await conn.commit()
            await store.create_task(TaskCreate(task_id="dead", type="scan"))
            claimed = await store.claim_task("worker-a", TaskQuery(type="scan"), lease_seconds=1)
            assert claimed is not None and claimed.lease_expires_at is not None
            await store.mark_claimed_task_execution_started(
                "dead", "worker-a", claimed.lease_expires_at
            )
            await store.create_task(TaskCreate(task_id="live", type="scan"))
            live = await store.claim_task("worker-b", TaskQuery(type="scan"), lease_seconds=300)
            assert live is not None and live.id == "live"
            # Postgres decides lease expiry by its own clock, so wait it out.
            await asyncio.sleep(1.2)
            assert await store.reclaim_expired(query=TaskQuery(type="scan")) == []

            fenced = await store.list_tasks(TaskQuery(fenced_after_worker_loss=True))
            assert _ids(fenced) == ["dead"]
            assert is_fenced_after_worker_loss(fenced[0])
            assert _ids(await store.list_tasks(TaskQuery(fenced_after_worker_loss=False))) == [
                "live"
            ]
        finally:
            await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("value", [0, 1, "true"])
def test_fenced_filter_requires_a_strict_boolean_and_survives_copies(value: object) -> None:
    from pydantic import ValidationError

    from cayu.tasks.queries import copy_task_query

    with pytest.raises(ValidationError):
        TaskQuery(fenced_after_worker_loss=value)
    assert (
        copy_task_query(TaskQuery(fenced_after_worker_loss=False)).fenced_after_worker_loss is False
    )


def test_settlement_and_warnings_reach_tasks_past_the_first_page(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import cayu.tasks.abandoned as abandoned

    caplog.set_level(logging.WARNING, logger="cayu.tasks.abandoned")
    monkeypatch.setattr(abandoned, "_SCAN_PAGE", 2)
    monkeypatch.setattr(abandoned, "_SCAN_MAX", 4)

    async def scenario() -> None:
        clock = _Clock()
        store = InMemoryTaskStore(ownership_clock=clock)
        started = [await _started_claim(store, f"own-{index}", "worker-new") for index in range(5)]
        started.append(await _started_claim(store, "zz-dead", "worker-old"))
        await _fence_after_worker_loss(store, clock, started)
        # Ineligible fences fill the first pages; the standalone call reads past them.
        [settled] = await reconcile_abandoned_tasks(
            store,
            live_worker_ids=["worker-new"],
            validator_id="service.single-worker",
            validator_version="1",
            query=TaskQuery(type="scan", limit=2),
        )
        assert settled.id == "zz-dead"

        await store.create_task(TaskCreate(task_id="zz-dead-2", type="scan"))
        late = await store.claim_task("worker-old", TaskQuery(type="scan"), lease_seconds=1)
        assert late is not None and late.lease_expires_at is not None
        await store.mark_claimed_task_execution_started(
            late.id, "worker-old", late.lease_expires_at
        )
        clock.value = late.lease_expires_at + timedelta(seconds=1)
        await store.reclaim_expired(query=TaskQuery(type="scan"))
        clock.value += timedelta(seconds=600)
        monitor = abandoned.FencedTaskMonitor(
            store,
            worker_id="worker-new",
            query=TaskQuery(type="scan"),
            reconcile_abandoned_after_s=0,
        )
        # The first scan stops after four tasks; the next one continues there.
        assert await monitor.scan() == []
        [later] = await monitor.scan()
        assert later.id == "zz-dead-2" and later.status is TaskStatus.CANCELLED
        await monitor.scan()

    asyncio.run(scenario())
    own_warnings = _warnings(caplog, "this worker's own id")
    assert sorted(message.split()[1] for message in own_warnings) == [
        f"own-{index}" for index in range(5)
    ]


def test_task_index_queries_refuse_the_fenced_filter() -> None:
    from cayu.tasks.index import TaskIndexQuery

    with pytest.raises(ValueError, match="fenced_after_worker_loss"):
        TaskIndexQuery(filters=TaskQuery(fenced_after_worker_loss=True))


def test_the_declared_wait_survives_scans_that_each_read_part_of_the_backlog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.tasks.abandoned as abandoned

    monkeypatch.setattr(abandoned, "_SCAN_PAGE", 2)
    monkeypatch.setattr(abandoned, "_SCAN_MAX", 4)

    async def scenario() -> None:
        clock = _Clock()
        store = InMemoryTaskStore(ownership_clock=clock)
        started = [
            await _started_claim(store, f"dead-{index}", "worker-old") for index in range(10)
        ]
        await _fence_after_worker_loss(store, clock, started)
        elapsed = [0.0]
        monitor = abandoned.FencedTaskMonitor(
            store,
            worker_id="worker-new",
            query=TaskQuery(type="scan"),
            reconcile_abandoned_after_s=100,
            clock=lambda: elapsed[0],
        )
        # Three scans see the whole backlog of ten, four at a time.
        for _ in range(3):
            assert await monitor.scan() == []
        elapsed[0] = 100.0
        settled: list[Task] = []
        for _ in range(4):
            settled.extend(await monitor.scan())
        assert sorted(task.id for task in settled) == [f"dead-{index}" for index in range(10)]
        assert await store.list_tasks(TaskQuery(fenced_after_worker_loss=True)) == []

    asyncio.run(scenario())
