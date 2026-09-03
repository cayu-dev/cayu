from __future__ import annotations

import asyncio
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError
from tests.core.task_invocation_fixtures import unattributed_task_invocation
from tests.core.task_store_conformance import assert_task_store_time_conformance

from cayu import (
    CayuApp,
    InMemoryTaskStore,
    SQLiteTaskStore,
    Task,
    TaskClaimLost,
    TaskCreate,
    TaskOrder,
    TaskQuery,
    TaskStatus,
)


class _MutableClock:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value


def test_task_availability_requires_aware_time_and_normalizes_to_utc() -> None:
    local_time = datetime(
        2026,
        8,
        8,
        9,
        30,
        tzinfo=timezone(timedelta(hours=5, minutes=30)),
    )

    request = TaskCreate(type="scheduled", available_at=local_time)
    task = Task(
        type="scheduled",
        available_at=local_time,
        invocation=unattributed_task_invocation(),
    )

    expected = datetime(2026, 8, 8, 4, 0, tzinfo=UTC)
    assert request.available_at == expected
    assert task.available_at == expected

    with pytest.raises(ValidationError, match="available_at must be timezone-aware"):
        TaskCreate(type="scheduled", available_at=datetime(2026, 8, 8, 4, 0))
    with pytest.raises(ValidationError, match="available_at must be timezone-aware"):
        Task(
            type="scheduled",
            available_at=datetime(2026, 8, 8, 4, 0),
            invocation=unattributed_task_invocation(),
        )


def test_in_memory_claims_only_tasks_available_at_the_store_clock() -> None:
    async def run() -> None:
        boundary = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
        clock = _MutableClock(boundary - timedelta(microseconds=1))
        store = InMemoryTaskStore(clock=clock)

        future = await store.create_task(
            TaskCreate(
                task_id="future-first",
                type="scheduled",
                available_at=boundary,
            )
        )
        await store.create_task(TaskCreate(task_id="immediate-second", type="scheduled"))

        immediate = await store.claim_task("worker-before")
        assert immediate is not None
        assert immediate.id == "immediate-second"
        assert await store.claim_task("worker-still-before") is None

        clock.value = boundary
        due = await store.claim_task("worker-at-boundary")
        assert due is not None
        assert due.id == future.id
        assert due.available_at == boundary

        await store.create_task(
            TaskCreate(
                task_id="after-boundary",
                type="scheduled",
                available_at=boundary,
            )
        )
        clock.value = boundary + timedelta(microseconds=1)
        after = await store.claim_task("worker-after")
        assert after is not None
        assert after.id == "after-boundary"

    asyncio.run(run())


def test_injected_availability_clock_does_not_expire_new_task_leases(tmp_path) -> None:
    async def run() -> None:
        boundary = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
        clock = _MutableClock(boundary)
        stores = (
            InMemoryTaskStore(clock=clock),
            SQLiteTaskStore(tmp_path / "lease-clock.sqlite", clock=clock),
        )
        try:
            for index, store in enumerate(stores):
                task_id = f"lease-clock-{index}"
                await store.create_task(
                    TaskCreate(
                        task_id=task_id,
                        type="scheduled",
                        available_at=boundary,
                    )
                )
                claimed = await store.claim_task(f"worker-{index}")
                assert claimed is not None
                released = await store.release_task(
                    task_id,
                    f"worker-{index}",
                    lease_expires_at=claimed.lease_expires_at,
                )
                assert released.status is TaskStatus.PENDING
        finally:
            await stores[1].close()

    asyncio.run(run())


def test_sqlite_heartbeat_samples_ownership_time_after_writer_lock(tmp_path) -> None:
    async def run() -> None:
        boundary = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
        ownership_clock = _MutableClock(boundary)
        path = tmp_path / "heartbeat-lock-clock.sqlite"
        store = SQLiteTaskStore(path, ownership_clock=ownership_clock)
        blocker: sqlite3.Connection | None = None
        release_timer: threading.Timer | None = None
        try:
            await store.create_task(TaskCreate(task_id="locked-heartbeat", type="scheduled"))
            claimed = await store.claim_task("worker", lease_seconds=1)
            assert claimed is not None

            blocker = sqlite3.connect(path, timeout=5, check_same_thread=False)
            blocker.execute("PRAGMA busy_timeout = 5000")
            blocker.execute("BEGIN IMMEDIATE")

            def release_writer() -> None:
                ownership_clock.value = boundary + timedelta(seconds=2)
                assert blocker is not None
                blocker.commit()

            release_timer = threading.Timer(0.05, release_writer)
            release_timer.start()
            with pytest.raises(TaskClaimLost, match="lease.*expired"):
                await store.heartbeat(
                    "locked-heartbeat",
                    "worker",
                    lease_expires_at=claimed.lease_expires_at,
                    extend_seconds=30,
                )
        finally:
            if release_timer is not None:
                release_timer.join(timeout=5)
            if blocker is not None:
                blocker.close()
            await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_task_ownership_uses_the_store_clock(backend: str, tmp_path) -> None:
    async def run() -> None:
        boundary = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
        availability_clock = _MutableClock(boundary)
        ownership_clock = _MutableClock(boundary)
        if backend == "memory":
            store: InMemoryTaskStore | SQLiteTaskStore = InMemoryTaskStore(
                clock=availability_clock,
                ownership_clock=ownership_clock,
            )
            contender_store: SQLiteTaskStore | None = None
        else:
            path = tmp_path / "authoritative-task-clock.sqlite"
            store = SQLiteTaskStore(
                path,
                clock=availability_clock,
                ownership_clock=ownership_clock,
            )
            contender_store = SQLiteTaskStore(
                path,
                clock=availability_clock,
                ownership_clock=ownership_clock,
            )
        try:
            await assert_task_store_time_conformance(
                store,
                initial_time=boundary,
                set_evidence_time=lambda value: setattr(
                    availability_clock,
                    "value",
                    value,
                ),
                set_ownership_time=lambda value: setattr(
                    ownership_clock,
                    "value",
                    value,
                ),
                contender_store=contender_store,
            )
        finally:
            if contender_store is not None:
                await contender_store.close()
            if isinstance(store, SQLiteTaskStore):
                await store.close()

    asyncio.run(run())


def test_app_rejects_delayed_tasks_before_calling_an_unsupported_custom_store() -> None:
    class UnsupportedDelayedAvailabilityStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True
        supports_delayed_availability = False

        def __init__(self) -> None:
            super().__init__()
            self.create_called = False

        async def create_task(self, request: TaskCreate) -> Task:
            self.create_called = True
            return await super().create_task(request.model_copy(update={"available_at": None}))

    async def run() -> None:
        store = UnsupportedDelayedAvailabilityStore()
        app = CayuApp(task_store=store)

        immediate = await app.create_task(TaskCreate(task_id="immediate", type="scheduled"))
        assert immediate.available_at is None

        store.create_called = False
        with pytest.raises(
            NotImplementedError,
            match="does not support delayed task availability",
        ):
            await app.create_task(
                TaskCreate(
                    task_id="unsupported-delayed",
                    type="scheduled",
                    available_at=datetime(2026, 8, 8, 12, 0, tzinfo=UTC),
                )
            )
        assert store.create_called is False
        assert await store.load_task("unsupported-delayed") is None

    asyncio.run(run())


def test_operational_snapshot_distinguishes_claimable_and_scheduled_pending() -> None:
    async def run() -> None:
        boundary = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
        clock = _MutableClock(boundary - timedelta(microseconds=1))
        store = InMemoryTaskStore(clock=clock)

        await store.create_task(TaskCreate(task_id="claimable", type="scheduled"))
        await store.create_task(
            TaskCreate(
                task_id="future",
                type="scheduled",
                available_at=boundary,
            )
        )
        await store.create_task(
            TaskCreate(
                task_id="session-bound",
                type="scheduled",
                session_id="session-1",
            )
        )

        before = await store.aggregate_operational_snapshot()
        assert before.counts_by_status.pending == 3
        assert before.claimable_pending_count == 1
        assert before.scheduled_pending_count == 1

        clock.value = boundary
        at_boundary = await store.aggregate_operational_snapshot()
        assert at_boundary.claimable_pending_count == 2
        assert at_boundary.scheduled_pending_count == 0

    asyncio.run(run())


def test_sqlite_persists_availability_and_claims_once_at_exact_boundary(tmp_path) -> None:
    async def run() -> None:
        boundary = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
        clock = _MutableClock(boundary - timedelta(microseconds=1))
        path = tmp_path / "tasks.sqlite"
        creator = SQLiteTaskStore(path, clock=clock)
        try:
            await creator.create_task(
                TaskCreate(
                    task_id="durable-future",
                    type="scheduled",
                    available_at=boundary,
                )
            )
            await creator.create_task(
                TaskCreate(
                    task_id="after-boundary",
                    type="scheduled",
                    available_at=boundary + timedelta(microseconds=1),
                )
            )
        finally:
            await creator.close()

        first = SQLiteTaskStore(path, clock=clock)
        second = SQLiteTaskStore(path, clock=clock)
        try:
            loaded = await first.load_task("durable-future")
            assert loaded is not None
            assert loaded.available_at == boundary
            before = await first.aggregate_operational_snapshot()
            assert before.claimable_pending_count == 0
            assert before.scheduled_pending_count == 2
            assert await first.claim_task("worker-before") is None

            clock.value = boundary
            at_boundary = await first.aggregate_operational_snapshot()
            assert at_boundary.claimable_pending_count == 1
            assert at_boundary.scheduled_pending_count == 1
            claims = await asyncio.gather(
                first.claim_task("worker-one"),
                second.claim_task("worker-two"),
            )
            winners = [claim for claim in claims if claim is not None]
            assert len(winners) == 1
            assert winners[0].id == "durable-future"
            assert winners[0].available_at == boundary

            clock.value = boundary + timedelta(microseconds=1)
            after = await second.claim_task("worker-after")
            assert after is not None
            assert after.id == "after-boundary"
        finally:
            await first.close()
            await second.close()

    asyncio.run(run())


def test_sqlite_operational_snapshot_captures_time_after_lock_ownership(tmp_path) -> None:
    class _CountingClock(_MutableClock):
        def __init__(self, value: datetime) -> None:
            super().__init__(value)
            self.calls = 0

        def __call__(self) -> datetime:
            self.calls += 1
            return super().__call__()

    async def run() -> None:
        boundary = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
        clock = _CountingClock(boundary - timedelta(microseconds=1))
        store = SQLiteTaskStore(tmp_path / "snapshot-clock.sqlite", clock=clock)
        try:
            await store.create_task(
                TaskCreate(
                    task_id="crosses-boundary",
                    type="scheduled",
                    available_at=boundary,
                )
            )
            clock.calls = 0

            async with store._lock:
                snapshot_task = asyncio.create_task(store.aggregate_operational_snapshot())
                await asyncio.sleep(0)
                assert snapshot_task.done() is False
                assert clock.calls == 0
                clock.value = boundary

            snapshot = await snapshot_task
            assert clock.calls == 1
            assert snapshot.as_of == boundary
            assert snapshot.claimable_pending_count == 1
            assert snapshot.scheduled_pending_count == 0
        finally:
            await store.close()

    asyncio.run(run())


def test_sqlite_operational_snapshot_captures_time_after_executor_admission(tmp_path) -> None:
    class _CountingClock(_MutableClock):
        def __init__(self, value: datetime) -> None:
            super().__init__(value)
            self.calls = 0

        def __call__(self) -> datetime:
            self.calls += 1
            return super().__call__()

    async def run(executor: ThreadPoolExecutor) -> None:
        boundary = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
        clock = _CountingClock(boundary - timedelta(microseconds=1))
        path = tmp_path / "executor-snapshot-clock.sqlite"
        snapshot_store = SQLiteTaskStore(path, clock=clock)
        concurrent_store = SQLiteTaskStore(path, clock=clock)
        worker_started = threading.Event()
        release_worker = threading.Event()
        blocker = None
        snapshot_task = None
        try:
            loop = asyncio.get_running_loop()
            loop.set_default_executor(executor)

            def occupy_executor() -> None:
                worker_started.set()
                if not release_worker.wait(timeout=5):
                    raise TimeoutError("executor blocker was not released")

            blocker = loop.run_in_executor(None, occupy_executor)
            while not worker_started.is_set():
                await asyncio.sleep(0)

            clock.calls = 0
            snapshot_task = asyncio.create_task(snapshot_store.aggregate_operational_snapshot())
            await asyncio.sleep(0)
            assert snapshot_task.done() is False
            assert clock.calls == 0

            clock.value = boundary
            await concurrent_store.create_task(
                TaskCreate(
                    task_id="committed-while-snapshot-queued",
                    type="scheduled",
                    available_at=boundary,
                )
            )
            release_worker.set()
            await blocker

            snapshot = await snapshot_task
            assert clock.calls >= 1
            assert snapshot.as_of == boundary
            assert snapshot.total_count == 1
            assert snapshot.claimable_pending_count == 1
            assert snapshot.scheduled_pending_count == 0
        finally:
            release_worker.set()
            if blocker is not None:
                await blocker
            if snapshot_task is not None and not snapshot_task.done():
                await snapshot_task
            await concurrent_store.close()
            await snapshot_store.close()

    with ThreadPoolExecutor(max_workers=1) as executor:
        asyncio.run(run(executor))


def test_task_query_pagination_keeps_immediate_and_future_tasks_visible(tmp_path) -> None:
    async def run() -> None:
        boundary = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
        clock = _MutableClock(boundary - timedelta(days=1))
        stores = (
            InMemoryTaskStore(clock=clock),
            SQLiteTaskStore(tmp_path / "pagination.sqlite", clock=clock),
        )
        try:
            for store in stores:
                await store.create_task(
                    TaskCreate(
                        task_id="future-first",
                        type="scheduled",
                        available_at=boundary,
                    )
                )
                await store.create_task(TaskCreate(task_id="immediate-second", type="scheduled"))
                await store.create_task(
                    TaskCreate(
                        task_id="future-third",
                        type="scheduled",
                        available_at=boundary + timedelta(hours=1),
                    )
                )

                pages = [
                    await store.list_tasks(
                        TaskQuery(
                            order_by=TaskOrder.CREATED_AT_ASC,
                            limit=1,
                            offset=offset,
                        )
                    )
                    for offset in range(3)
                ]
                assert [page[0].id for page in pages] == [
                    "future-first",
                    "immediate-second",
                    "future-third",
                ]
        finally:
            await stores[1].close()

    asyncio.run(run())
