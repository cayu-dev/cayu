"""Loss-tolerant task-admission wakeup tests."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import process_time

import pytest

from cayu import (
    CayuApp,
    DurableWorkerMetrics,
    InMemoryTaskStore,
    SQLiteTaskStore,
    Task,
    TaskCreate,
    TaskQuery,
    TaskRetryAttemptDisposition,
    TaskRetryPolicy,
    TaskRetrySettlementRequest,
    run_task_worker,
)


class _ObservedInMemoryTaskStore(InMemoryTaskStore):
    verified_work_mutations_are_cancellation_quiescent = True

    def __init__(self) -> None:
        super().__init__()
        self.empty_claim = asyncio.Event()
        self.claim_calls = 0

    async def claim_task(
        self,
        worker_id: str,
        query: TaskQuery | None = None,
        *,
        lease_seconds: int = 300,
    ) -> Task | None:
        self.claim_calls += 1
        claimed = await super().claim_task(
            worker_id,
            query,
            lease_seconds=lease_seconds,
        )
        if claimed is None:
            self.empty_claim.set()
        return claimed


class _ObservedSQLiteTaskStore(SQLiteTaskStore):
    verified_work_mutations_are_cancellation_quiescent = True

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.empty_claim = asyncio.Event()
        self.claim_calls = 0

    async def claim_task(
        self,
        worker_id: str,
        query: TaskQuery | None = None,
        *,
        lease_seconds: int = 300,
    ) -> Task | None:
        self.claim_calls += 1
        claimed = await super().claim_task(
            worker_id,
            query,
            lease_seconds=lease_seconds,
        )
        if claimed is None:
            self.empty_claim.set()
        return claimed


async def _complete_handler(
    app: CayuApp,
    task: Task,
    worker_id: str,
) -> None:
    assert app.task_store is not None
    assert task.lease_expires_at is not None
    await app.task_store.complete_task(
        task.id,
        {"ok": True},
        worker_id=worker_id,
        lease_expires_at=task.lease_expires_at,
    )


async def _wait_for_subscribers(store: InMemoryTaskStore | SQLiteTaskStore, count: int) -> None:
    async with asyncio.timeout(1):
        while store._task_admission_wakeup_broker.subscriber_count != count:
            await asyncio.sleep(0)


@pytest.mark.anyio
async def test_builtin_stores_publish_only_after_creation_commit(tmp_path: Path) -> None:
    observed: list[str] = []

    class OrderedInMemoryStore(InMemoryTaskStore):
        def _publish_task_admission_wakeup(self, task: Task, *, now: datetime) -> None:
            assert not self._lock.locked()
            assert task.id in self._tasks
            observed.append("memory")
            super()._publish_task_admission_wakeup(task, now=now)

    class OrderedSQLiteStore(SQLiteTaskStore):
        def _publish_task_admission_wakeup(self, task: Task, *, now: datetime) -> None:
            assert not self._lock.locked()
            assert not self._connection.in_transaction
            assert self._load_task_unlocked(task.id) is not None
            observed.append("sqlite")
            super()._publish_task_admission_wakeup(task, now=now)

    memory = OrderedInMemoryStore()
    sqlite = OrderedSQLiteStore(tmp_path / "commit-before-wake.sqlite")
    try:
        await memory.create_task(TaskCreate(task_id="memory-task", type="job"))
        await sqlite.create_task(TaskCreate(task_id="sqlite-task", type="job"))
    finally:
        await sqlite.close()

    assert observed == ["memory", "sqlite"]


@pytest.mark.anyio
async def test_matching_admission_wakes_idle_task_worker_before_long_poll() -> None:
    store = _ObservedInMemoryTaskStore()
    app = CayuApp(task_store=store, enable_logging=False)
    handled = asyncio.Event()
    metrics = DurableWorkerMetrics()

    async def handler(app: CayuApp, task: Task, worker_id: str) -> None:
        await _complete_handler(app, task, worker_id)
        handled.set()

    worker = asyncio.create_task(
        run_task_worker(
            app,
            store,
            handler,
            worker_id="wake-worker",
            query=TaskQuery(type="job"),
            poll_interval_s=10.0,
            metrics=metrics,
            reclaim=False,
            recover_interrupted_handoffs=False,
            max_tasks=1,
        )
    )
    await asyncio.wait_for(store.empty_claim.wait(), timeout=1)

    started_at = asyncio.get_running_loop().time()
    await store.create_task(TaskCreate(task_id="prompt-job", type="job"))
    await asyncio.wait_for(handled.wait(), timeout=0.5)

    assert asyncio.get_running_loop().time() - started_at < 0.5
    assert await worker == 1
    snapshot = metrics.snapshot()
    assert snapshot.wake_hints_followed_by_successful_claims == 1
    assert snapshot.admission_to_claim_latency_samples == 1
    assert snapshot.admission_to_claim_latency_max_s < 0.5
    assert store._task_admission_wakeup_broker.subscriber_count == 0


@pytest.mark.anyio
@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
async def test_hundred_idle_task_workers_meet_economics_budget(
    store_kind: str,
    tmp_path: Path,
) -> None:
    database = tmp_path / "hundred-worker-economics.sqlite"
    store = (
        _ObservedInMemoryTaskStore()
        if store_kind == "memory"
        else _ObservedSQLiteTaskStore(database)
    )
    producer = store if store_kind == "memory" else SQLiteTaskStore(database)
    app = CayuApp(task_store=store, enable_logging=False)
    stop = asyncio.Event()
    handled = asyncio.Event()
    metrics = DurableWorkerMetrics(configured_handler_capacity=100)

    async def handler(app: CayuApp, task: Task, worker_id: str) -> None:
        await _complete_handler(app, task, worker_id)
        handled.set()

    workers = [
        asyncio.create_task(
            run_task_worker(
                app,
                store,
                handler,
                worker_id=f"pooled-worker-{index}",
                query=TaskQuery(type="job"),
                poll_interval_s=0.5,
                minimum_idle_delay_s=0.01,
                maximum_idle_delay_s=0.05,
                idle_jitter_ratio=0.0,
                metrics=metrics,
                reclaim=False,
                recover_interrupted_handoffs=False,
                stop=stop,
                max_tasks=1,
            )
        )
        for index in range(100)
    ]
    try:
        await _wait_for_subscribers(store, 100)
        await asyncio.wait_for(store.empty_claim.wait(), timeout=1)
        cpu_started = process_time()
        await asyncio.sleep(0.15)
        idle_cpu_s = process_time() - cpu_started

        assert 2 <= store.claim_calls <= 10
        assert idle_cpu_s <= 0.10

        await producer.create_task(TaskCreate(task_id="pooled-job", type="job"))
        await asyncio.wait_for(handled.wait(), timeout=0.5)
    finally:
        stop.set()
        handled_counts = await asyncio.gather(*workers)
        if producer is not store:
            await producer.close()
        close = getattr(store, "close", None)
        if close is not None:
            await close()

    assert sum(handled_counts) == 1
    snapshot = metrics.snapshot()
    assert snapshot.configured_handler_capacity == 100
    assert snapshot.maximum_active_pollers == 1
    assert snapshot.maximum_active_handlers == 1
    assert snapshot.claim_attempts == store.claim_calls
    assert snapshot.successful_claims == 1
    assert snapshot.admission_to_claim_latency_samples == 1
    assert snapshot.admission_to_claim_latency_max_s <= 0.5
    if store_kind == "memory":
        assert snapshot.wake_hints_received == 1
        assert snapshot.wake_hints_accepted == 1
        assert snapshot.wake_hints_ignored == 0
        assert snapshot.wake_hints_followed_by_successful_claims == 1
    else:
        assert snapshot.wake_hints_received == 0
        assert snapshot.fallback_poll_activations >= 1
    assert store._task_admission_wakeup_broker.subscriber_count == 0
    groups = store._durable_worker_poller_groups
    assert groups == {}


@pytest.mark.anyio
async def test_admission_between_empty_claim_and_wait_is_not_lost() -> None:
    class RacingStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True
        injected = False

        async def claim_task(
            self,
            worker_id: str,
            query: TaskQuery | None = None,
            *,
            lease_seconds: int = 300,
        ) -> Task | None:
            claimed = await super().claim_task(
                worker_id,
                query,
                lease_seconds=lease_seconds,
            )
            if claimed is None and not self.injected:
                self.injected = True
                await self.create_task(TaskCreate(task_id="raced-job", type="job"))
            return claimed

    store = RacingStore()
    app = CayuApp(task_store=store, enable_logging=False)

    worker = asyncio.create_task(
        run_task_worker(
            app,
            store,
            _complete_handler,
            worker_id="race-worker",
            query=TaskQuery(type="job"),
            poll_interval_s=10.0,
            reclaim=False,
            recover_interrupted_handoffs=False,
            max_tasks=1,
        )
    )

    assert await asyncio.wait_for(worker, timeout=0.5) == 1
    completed = await store.load_task("raced-job")
    assert completed is not None and completed.status.value == "completed"


@pytest.mark.anyio
async def test_nonmatching_and_future_admissions_do_not_wake_worker() -> None:
    store = _ObservedInMemoryTaskStore()
    app = CayuApp(task_store=store, enable_logging=False)
    stop = asyncio.Event()
    worker = asyncio.create_task(
        run_task_worker(
            app,
            store,
            _complete_handler,
            worker_id="filtered-worker",
            query=TaskQuery(type="target"),
            poll_interval_s=10.0,
            reclaim=False,
            recover_interrupted_handoffs=False,
            stop=stop,
        )
    )
    await asyncio.wait_for(store.empty_claim.wait(), timeout=1)

    await store.create_task(TaskCreate(task_id="other", type="other"))
    await store.create_task(
        TaskCreate(
            task_id="future-target",
            type="target",
            available_at=datetime.now(UTC) + timedelta(hours=1),
        )
    )
    await asyncio.sleep(0.05)

    assert store.claim_calls == 1
    stop.set()
    assert await asyncio.wait_for(worker, timeout=0.5) == 0
    assert store._task_admission_wakeup_broker.subscriber_count == 0


@pytest.mark.anyio
async def test_admission_hint_matching_honors_parent_and_agent_filters() -> None:
    store = InMemoryTaskStore()
    parent = await store.create_task(TaskCreate(task_id="parent", type="workflow"))
    wakeup = await store._task_admission_wakeup(
        (
            TaskQuery(
                type="job",
                parent_task_id=parent.id,
                assigned_agent_name="agent-a",
            ),
        )
    )
    assert wakeup is not None
    try:
        hinted = asyncio.create_task(wakeup.wait(10.0, None))
        await store.create_task(
            TaskCreate(
                task_id="wrong-agent",
                type="job",
                parent_task_id=parent.id,
                assigned_agent_name="agent-b",
            )
        )
        await asyncio.sleep(0)
        assert not hinted.done()

        await store.create_task(
            TaskCreate(
                task_id="matching-agent",
                type="job",
                parent_task_id=parent.id,
                assigned_agent_name="agent-a",
            )
        )
        assert await asyncio.wait_for(hinted, timeout=0.5) is False
    finally:
        wakeup.close()


@pytest.mark.anyio
@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
async def test_immediate_retry_successor_wakes_once_after_settlement_commit(
    store_kind: str,
    tmp_path: Path,
) -> None:
    store = (
        InMemoryTaskStore()
        if store_kind == "memory"
        else SQLiteTaskStore(tmp_path / "retry-admission.sqlite")
    )
    wakeup = None
    replay_wait = None
    try:
        await store.create_task(
            TaskCreate(
                task_id="retry-first",
                type="job",
                retry_policy=TaskRetryPolicy(
                    max_attempts=2,
                    initial_backoff_seconds=0,
                ),
            )
        )
        claimed = await store.claim_task("retry-worker", TaskQuery(type="job"))
        assert claimed is not None and claimed.retry_series is not None
        assert claimed.lease_expires_at is not None
        request = TaskRetrySettlementRequest(
            task_id=claimed.id,
            worker_id="retry-worker",
            lease_expires_at=claimed.lease_expires_at,
            idempotency_key="retry-once",
            causal_budget_id=claimed.retry_series.causal_budget_id,
            disposition=TaskRetryAttemptDisposition.RETRYABLE_FAILURE,
            error={"code": "temporary"},
        )
        wakeup = await store._task_admission_wakeup((TaskQuery(type="job"),))
        assert wakeup is not None
        hinted = asyncio.create_task(wakeup.wait(10.0, None))
        await asyncio.sleep(0)

        receipt = await store.settle_task_retry_attempt(request)

        assert receipt.successor is not None
        assert await asyncio.wait_for(hinted, timeout=0.5) is False

        replay_wait = asyncio.create_task(wakeup.wait(10.0, None))
        await asyncio.sleep(0)
        assert await store.settle_task_retry_attempt(request) == receipt
        await asyncio.sleep(0.05)
        assert not replay_wait.done()
    finally:
        if replay_wait is not None:
            replay_wait.cancel()
            await asyncio.gather(replay_wait, return_exceptions=True)
        if wakeup is not None:
            wakeup.close()
        close = getattr(store, "close", None)
        if close is not None:
            await close()


@pytest.mark.anyio
async def test_duplicate_and_stale_hints_never_duplicate_claim_authority() -> None:
    store = InMemoryTaskStore()
    wakeup = await store._task_admission_wakeup((TaskQuery(type="job"),))
    assert wakeup is not None
    try:
        created = await store.create_task(TaskCreate(task_id="coalesced-job", type="job"))
        store._publish_task_admission_wakeup(created, now=datetime.now(UTC))
        store._publish_task_admission_wakeup(created, now=datetime.now(UTC))
        assert await wakeup.wait(0.5, None) is False

        claimed = await store.claim_task("winner", TaskQuery(type="job"))
        assert claimed is not None and claimed.id == created.id
        await store.complete_task(
            claimed.id,
            {"ok": True},
            worker_id="winner",
            lease_expires_at=claimed.lease_expires_at,
        )
        store._publish_task_admission_broadcast()
        assert await wakeup.wait(0.5, None) is False
        assert await store.claim_task("late-worker", TaskQuery(type="job")) is None
    finally:
        wakeup.close()


@pytest.mark.anyio
async def test_one_admission_wakes_one_of_multiple_workers_without_duplicate_claim() -> None:
    store = InMemoryTaskStore()
    app = CayuApp(task_store=store, enable_logging=False)
    stop = asyncio.Event()
    handled: list[str] = []

    async def handler(app: CayuApp, task: Task, worker_id: str) -> None:
        handled.append(task.id)
        await _complete_handler(app, task, worker_id)
        stop.set()

    workers = [
        asyncio.create_task(
            run_task_worker(
                app,
                store,
                handler,
                worker_id=f"worker-{index}",
                query=TaskQuery(type="job"),
                poll_interval_s=10.0,
                reclaim=False,
                recover_interrupted_handoffs=False,
                stop=stop,
                max_tasks=1,
            )
        )
        for index in range(2)
    ]
    await _wait_for_subscribers(store, 2)
    await store.create_task(TaskCreate(task_id="single-job", type="job"))

    assert sum(await asyncio.wait_for(asyncio.gather(*workers), timeout=1)) == 1
    assert handled == ["single-job"]
    assert store._task_admission_wakeup_broker.subscriber_count == 0


@pytest.mark.anyio
async def test_worker_cancellation_unregisters_admission_waiter() -> None:
    store = InMemoryTaskStore()
    app = CayuApp(task_store=store, enable_logging=False)
    worker = asyncio.create_task(
        run_task_worker(
            app,
            store,
            _complete_handler,
            worker_id="cancelled-worker",
            poll_interval_s=10.0,
            reclaim=False,
            recover_interrupted_handoffs=False,
        )
    )
    await _wait_for_subscribers(store, 1)

    worker.cancel()
    with pytest.raises(asyncio.CancelledError):
        await worker

    assert store._task_admission_wakeup_broker.subscriber_count == 0


@pytest.mark.anyio
async def test_sqlite_separate_store_instance_converges_through_polling(
    tmp_path: Path,
) -> None:
    database = tmp_path / "task-wakeup.sqlite"
    producer = SQLiteTaskStore(database)
    consumer = SQLiteTaskStore(database)
    app = CayuApp(task_store=consumer, enable_logging=False)
    try:
        worker = asyncio.create_task(
            run_task_worker(
                app,
                consumer,
                _complete_handler,
                worker_id="sqlite-worker",
                query=TaskQuery(type="job"),
                poll_interval_s=0.02,
                reclaim=False,
                recover_interrupted_handoffs=False,
                max_tasks=1,
            )
        )
        await _wait_for_subscribers(consumer, 1)
        await asyncio.sleep(0.03)
        await producer.create_task(TaskCreate(task_id="sqlite-job", type="job"))

        assert await asyncio.wait_for(worker, timeout=0.5) == 1
    finally:
        await producer.close()
        await consumer.close()
