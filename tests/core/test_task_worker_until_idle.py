from __future__ import annotations

import asyncio
import inspect
from datetime import UTC, datetime, timedelta

import pytest

from cayu import CayuApp
from cayu.storage.migrations import SchemaMode
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.storage.tasks_postgres import PostgresTaskStore
from cayu.tasks import (
    TaskCreate,
    TaskQuery,
    TaskSchedulePolicy,
    TaskScheduleWakeup,
    TaskStatus,
    TaskWorkerIdleResult,
    complete_managed_task,
    run_task_worker,
    run_task_worker_until_idle,
)
from cayu.tasks import worker as task_worker
from cayu.tasks.handoff import InterruptedTaskContinuationClaimPage
from cayu.tasks.memory import InMemoryTaskStore

# Long enough that a worker which waited for its next poll would miss the timeout.
_SLOW_POLL_S = 30.0
_EXIT_TIMEOUT_S = 10.0


class _ClaimRecordingStore(InMemoryTaskStore):
    verified_work_mutations_are_cancellation_quiescent = True

    def __init__(self) -> None:
        super().__init__()
        self.claims: list[str | None] = []
        self.empty_claim = asyncio.Event()

    async def claim_task(self, *args, **kwargs):
        task = await super().claim_task(*args, **kwargs)
        self.claims.append(None if task is None else task.id)
        if task is None:
            self.empty_claim.set()
        return task


def _store(backend: str, tmp_path, postgres_dsn):
    if backend == "memory":
        return InMemoryTaskStore()
    if backend == "sqlite":
        return SQLiteTaskStore(tmp_path / "tasks.sqlite")
    return PostgresTaskStore(postgres_dsn, schema_mode=SchemaMode.CREATE)


async def _until_idle(app, store, handler, task_type, **options) -> TaskWorkerIdleResult:
    return await asyncio.wait_for(
        run_task_worker_until_idle(
            app,
            store,
            handler,
            worker_id=f"{task_type}-worker",
            query=TaskQuery(type=task_type),
            poll_interval_s=_SLOW_POLL_S,
            **options,
        ),
        timeout=_EXIT_TIMEOUT_S,
    )


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_until_idle_runs_due_tasks_then_reports_the_next_wake(tmp_path, postgres_dsn, backend):
    async def run():
        store = _store(backend, tmp_path, postgres_dsn)
        task_type = f"until-idle-{backend}"
        try:
            app = CayuApp(task_store=store, enable_logging=False)
            now = datetime.now(UTC)
            later = now + timedelta(hours=1)
            expires = now + timedelta(hours=3)
            for index in range(2):
                await app.create_task(
                    TaskCreate(task_id=f"{task_type}-due-{index}", type=task_type)
                )
            await app.create_task(
                TaskCreate(
                    task_id=f"{task_type}-later",
                    type=task_type,
                    available_at=later,
                    schedule_policy=TaskSchedulePolicy(),
                )
            )
            await app.create_task(
                TaskCreate(
                    task_id=f"{task_type}-expiring",
                    type=task_type,
                    available_at=now + timedelta(hours=2),
                    schedule_policy=TaskSchedulePolicy(expires_at=expires),
                )
            )
            handled: list[str] = []

            async def handler(_app, task, worker_id):
                handled.append(task.id)
                await complete_managed_task(store, task, worker_id, {"ok": True})

            result = await _until_idle(app, store, handler, task_type)

            assert sorted(handled) == [f"{task_type}-due-0", f"{task_type}-due-1"]
            assert result == TaskWorkerIdleResult(
                handled=2,
                idle=True,
                next_available_at=later,
                next_expiry_at=expires,
                maintenance_required=False,
            )
            for task_id in (f"{task_type}-later", f"{task_type}-expiring"):
                task = await store.load_task(task_id)
                assert task is not None and task.status is TaskStatus.PENDING
        finally:
            if backend != "memory":
                await store.close()

    asyncio.run(run())


def test_until_idle_returns_at_once_when_nothing_is_due():
    async def run():
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)

        async def handler(_app, _task, _worker_id):
            raise AssertionError("nothing is due")

        result = await _until_idle(app, store, handler, "empty")

        assert result == TaskWorkerIdleResult(handled=0, idle=True)
        assert result.model_dump(mode="json") == {
            "handled": 0,
            "idle": True,
            "next_available_at": None,
            "next_expiry_at": None,
            "maintenance_required": False,
            "continuations_pending": False,
        }

    asyncio.run(run())


@pytest.mark.parametrize("concurrency", [1, 2])
def test_until_idle_waits_for_running_handlers_and_the_work_they_create(concurrency):
    async def run():
        store = _ClaimRecordingStore()
        app = CayuApp(task_store=store, enable_logging=False)
        await app.create_task(TaskCreate(task_id="parent", type="chain"))
        handled: list[str] = []

        async def handler(_app, task, worker_id):
            if task.id == "parent":
                if concurrency > 1:
                    # Another slot finds nothing while this handler still runs.
                    await asyncio.wait_for(store.empty_claim.wait(), timeout=_EXIT_TIMEOUT_S)
                await app.create_task(TaskCreate(task_id="child", type="chain"))
            handled.append(task.id)
            await complete_managed_task(store, task, worker_id, {"ok": True})

        result = await _until_idle(app, store, handler, "chain", concurrency=concurrency)

        assert handled == ["parent", "child"]
        assert result.handled == 2 and result.idle

    asyncio.run(run())


def test_until_idle_stops_at_max_tasks_without_claiming_idle():
    async def run():
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        for index in range(3):
            await app.create_task(TaskCreate(task_id=f"capped-{index}", type="capped"))

        async def handler(_app, task, worker_id):
            await complete_managed_task(store, task, worker_id, {"ok": True})

        result = await _until_idle(app, store, handler, "capped", max_tasks=2)

        assert result == TaskWorkerIdleResult(handled=2, idle=False)

    asyncio.run(run())


def test_until_idle_claims_again_once_when_schedule_maintenance_is_pending(monkeypatch):
    async def run():
        store = _ClaimRecordingStore()
        app = CayuApp(task_store=store, enable_logging=False)

        async def observe(query=None):
            # A sweep that never settles must not keep the worker alive.
            return TaskScheduleWakeup(as_of=datetime.now(UTC), maintenance_required=True)

        monkeypatch.setattr(store, "next_task_schedule_wakeup", observe)

        async def handler(_app, _task, _worker_id):
            raise AssertionError("nothing is due")

        result = await _until_idle(app, store, handler, "sweep")

        assert store.claims == [None, None]
        assert result == TaskWorkerIdleResult(handled=0, idle=True, maintenance_required=True)

    asyncio.run(run())


def test_until_idle_stops_when_asked_and_reports_it_was_not_idle():
    async def run():
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        stop = asyncio.Event()
        stop.set()

        async def handler(_app, _task, _worker_id):
            raise AssertionError("a stopped worker claims nothing")

        result = await _until_idle(app, store, handler, "stopped", stop=stop)

        assert result == TaskWorkerIdleResult(handled=0, idle=False)

    asyncio.run(run())


@pytest.mark.parametrize("concurrency", [1, 2])
@pytest.mark.parametrize("due_task", [False, True])
def test_until_idle_continues_handoffs_once_without_waiting_on_people(
    monkeypatch, concurrency, due_task
):
    """A handed-off session waiting on a person stays claimable on every scan."""

    async def run():
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        waiting = await app.create_task(TaskCreate(task_id="waiting-on-a-person", type="other"))
        if due_task:
            await app.create_task(TaskCreate(task_id="due", type="handoffs"))
        scans = 0

        async def continue_waiting_session(*_args, **_kwargs):
            nonlocal scans
            scans += 1
            await asyncio.sleep(0)  # A real scan reads the store.
            return InterruptedTaskContinuationClaimPage(
                task=waiting,
                next_after=(waiting.created_at, waiting.id),
                scanned_candidates=1,
                rejected_candidates=0,
                exhausted=True,
            )

        monkeypatch.setattr(
            task_worker, "_run_recovered_interrupted_task_handler", continue_waiting_session
        )

        async def handler(_app, task, worker_id):
            await complete_managed_task(store, task, worker_id, {"ok": True})

        async def continue_session(_app, _task, _worker_id):
            raise AssertionError("the scan is replaced in this test")

        result = await _until_idle(
            app,
            store,
            handler,
            "handoffs",
            recovered_interrupted_task_handler=continue_session,
            concurrency=concurrency,
        )

        assert scans == 1
        assert result.handled == 1 + due_task and result.idle
        # A handled task may have handed off its session after the one scan.
        assert result.continuations_pending is due_task

    asyncio.run(run())


def test_until_idle_continues_handoffs_that_recovery_finds_after_the_scan(monkeypatch):
    async def run():
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        pages = [
            task_worker._InterruptedHandoffRecoveryPage(
                recovered=0,
                next_after=(datetime.now(UTC), "first-page-end"),
                exhausted=False,
            ),
            task_worker._InterruptedHandoffRecoveryPage(
                recovered=1, next_after=None, exhausted=True
            ),
        ]
        recoveries = 0
        scans = 0

        async def recover(_app, _store, *, after, limit, stop):
            nonlocal recoveries
            recoveries += 1
            return pages[recoveries - 1]

        async def scan(*_args, **_kwargs):
            nonlocal scans
            scans += 1
            return InterruptedTaskContinuationClaimPage(
                scanned_candidates=0, rejected_candidates=0, exhausted=True
            )

        monkeypatch.setattr(task_worker, "_recover_expired_interrupted_task_handoffs", recover)
        monkeypatch.setattr(task_worker, "_run_recovered_interrupted_task_handler", scan)

        async def handler(_app, _task, _worker_id):
            raise AssertionError("nothing is due")

        async def continue_session(_app, _task, _worker_id):
            raise AssertionError("the scan is replaced in this test")

        result = await _until_idle(
            app,
            store,
            handler,
            "late-recovery",
            recovered_interrupted_task_handler=continue_session,
        )

        assert (recoveries, scans) == (2, 2)
        assert result.idle and not result.continuations_pending

    asyncio.run(run())


def test_until_idle_refuses_a_store_that_cannot_report_scheduled_times():
    class UnscheduledStore(InMemoryTaskStore):
        supports_task_scheduling = False

    async def run():
        store = UnscheduledStore()
        app = CayuApp(task_store=store, enable_logging=False)

        async def handler(_app, _task, _worker_id):
            raise AssertionError("the worker must not start")

        with pytest.raises(NotImplementedError, match="scheduled task times"):
            await _until_idle(app, store, handler, "unscheduled")

    asyncio.run(run())


def test_run_task_worker_keeps_until_idle_state_out_of_its_signature():
    assert all(not name.startswith("_") for name in inspect.signature(run_task_worker).parameters)


def test_until_idle_finishes_a_whole_expired_handoff_recovery_pass(monkeypatch):
    async def run():
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        pages = [
            task_worker._InterruptedHandoffRecoveryPage(
                recovered=0,
                next_after=(datetime.now(UTC), "first-page-end"),
                exhausted=False,
            ),
            task_worker._InterruptedHandoffRecoveryPage(
                recovered=0, next_after=None, exhausted=True
            ),
        ]
        afters = []

        async def recover(_app, _store, *, after, limit, stop):
            afters.append(after)
            return pages[len(afters) - 1]

        monkeypatch.setattr(task_worker, "_recover_expired_interrupted_task_handoffs", recover)

        async def handler(_app, _task, _worker_id):
            raise AssertionError("nothing is due")

        result = await _until_idle(app, store, handler, "recovery")

        assert afters == [None, pages[0].next_after]
        assert result.idle

    asyncio.run(run())


class _GatedEmptyClaimStore(InMemoryTaskStore):
    """Holds the first empty claim until the test releases it."""

    verified_work_mutations_are_cancellation_quiescent = True

    def __init__(self) -> None:
        super().__init__()
        self.empty_claim_waiting = asyncio.Event()
        self.release_empty_claim = asyncio.Event()
        self._gated = False

    async def claim_task(self, *args, **kwargs):
        task = await super().claim_task(*args, **kwargs)
        if task is None and not self._gated:
            self._gated = True
            self.empty_claim_waiting.set()
            await self.release_empty_claim.wait()
        return task


def test_until_idle_reobserves_after_another_slot_finishes_work(monkeypatch):
    """A slot's empty claim outlives the handler that schedules more work."""

    async def run():
        store = _GatedEmptyClaimStore()
        app = CayuApp(task_store=store, enable_logging=False)
        await app.create_task(TaskCreate(task_id="parent", type="follow-up"))
        later = datetime.now(UTC) + timedelta(hours=1)

        async def handler(_app, task, worker_id):
            # Another slot has observed the schedule and is claiming.
            await asyncio.wait_for(store.empty_claim_waiting.wait(), timeout=_EXIT_TIMEOUT_S)
            await app.create_task(
                TaskCreate(
                    task_id="follow-up-later",
                    type="follow-up",
                    available_at=later,
                    schedule_policy=TaskSchedulePolicy(),
                )
            )
            await complete_managed_task(store, task, worker_id, {"ok": True})

        run_worker_step = app._run_worker_step

        async def release_after_handled_step(step):
            result = await run_worker_step(step)
            if result.handled:
                # The handling slot leaves its step before the held claim returns.
                store.release_empty_claim.set()
            return result

        monkeypatch.setattr(app, "_run_worker_step", release_after_handled_step)

        result = await _until_idle(app, store, handler, "follow-up", concurrency=2)

        assert result.handled == 1 and result.idle
        assert result.next_available_at == later

    asyncio.run(run())


def test_until_idle_rescans_continuations_from_the_start_after_recovery(monkeypatch):
    """Recovery follows lease expiry; the scan follows creation, so it restarts."""

    async def run():
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        cursor = (datetime.now(UTC), "scanned-so-far")
        pages = [
            task_worker._InterruptedHandoffRecoveryPage(
                recovered=0, next_after=(datetime.now(UTC), "first-page-end"), exhausted=False
            ),
            task_worker._InterruptedHandoffRecoveryPage(
                recovered=1, next_after=None, exhausted=True
            ),
        ]
        recoveries = 0
        scanned_after = []

        async def recover(_app, _store, *, after, limit, stop):
            nonlocal recoveries
            recoveries += 1
            return pages[recoveries - 1]

        async def scan(*_args, after, **_kwargs):
            scanned_after.append(after)
            if len(scanned_after) == 1:
                return InterruptedTaskContinuationClaimPage(
                    next_after=cursor,
                    scanned_candidates=1,
                    rejected_candidates=0,
                    filtered_candidates=1,
                    exhausted=False,
                )
            return InterruptedTaskContinuationClaimPage(
                scanned_candidates=0, rejected_candidates=0, exhausted=True
            )

        monkeypatch.setattr(task_worker, "_recover_expired_interrupted_task_handoffs", recover)
        monkeypatch.setattr(task_worker, "_run_recovered_interrupted_task_handler", scan)

        async def handler(_app, _task, _worker_id):
            raise AssertionError("nothing is due")

        async def continue_session(_app, _task, _worker_id):
            raise AssertionError("the scan is replaced in this test")

        result = await _until_idle(
            app,
            store,
            handler,
            "rewind",
            recovered_interrupted_task_handler=continue_session,
        )

        # The handoff recovered on page two may sort before the scan's cursor.
        assert scanned_after == [None, None]
        assert result.idle

    asyncio.run(run())


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_until_idle_reclaims_every_expired_claim_before_returning(
    tmp_path, postgres_dsn, backend, monkeypatch
):
    async def run():
        monkeypatch.setattr(task_worker, "_RECLAIM_BATCH_SIZE", 2)
        store = _store(backend, tmp_path, postgres_dsn)
        task_type = f"expired-claims-{backend}"
        try:
            app = CayuApp(task_store=store, enable_logging=False)
            for index in range(3):
                await app.create_task(TaskCreate(task_id=f"{task_type}-{index}", type=task_type))
            for _ in range(3):
                # A worker that died before starting its handlers.
                claimed = await store.claim_task(
                    "lost-worker", TaskQuery(type=task_type), lease_seconds=1
                )
                assert claimed is not None
            await asyncio.sleep(1.2)
            handled: list[str] = []

            async def handler(_app, task, worker_id):
                handled.append(task.id)
                await complete_managed_task(store, task, worker_id, {"ok": True})

            result = await _until_idle(app, store, handler, task_type)

            assert sorted(handled) == [f"{task_type}-{index}" for index in range(3)]
            assert result.handled == 3 and result.idle
        finally:
            if backend != "memory":
                await store.close()

    asyncio.run(run())


def test_until_idle_keeps_reclaiming_when_reclaimed_tasks_cannot_run_yet(monkeypatch):
    """An empty claim right after a full reclaim batch doesn't end the run."""

    class FirstClaimBlockedStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True
        blocked = True

        async def claim_task(self, *args, **kwargs):
            if self.blocked:
                # As if the reclaimed tasks waited on a concurrency key.
                self.blocked = False
                return None
            return await super().claim_task(*args, **kwargs)

    async def run():
        monkeypatch.setattr(task_worker, "_RECLAIM_BATCH_SIZE", 2)
        store = FirstClaimBlockedStore()
        store.blocked = False
        app = CayuApp(task_store=store, enable_logging=False)
        for index in range(3):
            await app.create_task(TaskCreate(task_id=f"blocked-{index}", type="blocked"))
            assert await store.claim_task("lost-worker", TaskQuery(type="blocked"), lease_seconds=1)
        store.blocked = True
        await asyncio.sleep(1.2)

        async def handler(_app, task, worker_id):
            await complete_managed_task(store, task, worker_id, {"ok": True})

        result = await _until_idle(app, store, handler, "blocked")

        assert result.handled == 3 and result.idle

    asyncio.run(run())
