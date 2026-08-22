from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError
from tests.core.task_invocation_fixtures import unattributed_task_invocation

from cayu import (
    CayuApp,
    InMemoryTaskStore,
    SQLiteTaskStore,
    Task,
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
                released = await store.release_task(task_id, f"worker-{index}")
                assert released.status is TaskStatus.PENDING
        finally:
            await stores[1].close()

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
