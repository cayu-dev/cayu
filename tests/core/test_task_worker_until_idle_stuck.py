"""An until-idle run reports stuck work and fenced settle times, for its host or a person."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from tests.core.task_invocation_fixtures import task_backed_session_invocation

import cayu.external_wait_tasks as external_wait_tasks
from cayu import CayuApp
from cayu.runtime import ModelCompletionManualRecoveryRequired
from cayu.sessions.records import SessionStatus
from cayu.storage.migrations import SchemaMode
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.storage.tasks_postgres import PostgresTaskStore
from cayu.tasks import (
    TaskCreate,
    TaskQuery,
    TaskStatus,
    run_task_worker_until_idle,
)
from cayu.tasks import worker as task_worker
from cayu.tasks.abandoned import RECONCILE_ABANDONED_AFTER_ENV
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.records import Task
from cayu.tasks.store import TaskStore

LEASE_SECONDS = 30
# Long enough that a worker which waited for its next poll would miss the timeout.
_SLOW_POLL_S = 30.0
_EXIT_TIMEOUT_S = 10.0


@pytest.fixture(autouse=True)
def _no_ambient_declaration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(RECONCILE_ABANDONED_AFTER_ENV, raising=False)


class _Clock:
    """The store's ownership clock, an hour behind real time so wall-clock checks agree."""

    def __init__(self) -> None:
        self.value = datetime.now(UTC).replace(microsecond=0) - timedelta(hours=1)

    def __call__(self) -> datetime:
        return self.value


def _store(backend: str, tmp_path, clock: _Clock) -> TaskStore:
    if backend == "memory":
        return InMemoryTaskStore(clock=clock, ownership_clock=clock)
    return SQLiteTaskStore(tmp_path / "tasks.sqlite", clock=clock, ownership_clock=clock)


async def _claim(
    store: TaskStore, task_id: str, task_type: str, worker_id: str, lease: int
) -> Task:
    await store.create_task(TaskCreate(task_id=task_id, type=task_type))
    claimed = await store.claim_task(worker_id, TaskQuery(type=task_type), lease_seconds=lease)
    assert claimed is not None and claimed.id == task_id and claimed.lease_expires_at is not None
    return claimed


async def _attached_owner(store: TaskStore, task_id: str, worker_id: str, lease: int) -> Task:
    """A task running its session under a worker's claim, as a woken continuation is."""

    claimed = await _claim(store, task_id, "owner", worker_id, lease)
    return await store.attach_task(
        task_id,
        session_id=f"session-{task_id}",
        session_invocation=await task_backed_session_invocation(
            store, task_id, f"session-{task_id}"
        ),
        worker_id=worker_id,
        lease_expires_at=claimed.lease_expires_at,
    )


async def _fenced(
    store: TaskStore, clock: _Clock, task_id: str, worker_id: str = "worker-old"
) -> Task:
    """A started task whose worker was lost: its lease expired and reclamation fenced it."""

    claimed = await _claim(store, task_id, "scan", worker_id, LEASE_SECONDS)
    assert claimed.lease_expires_at is not None
    await store.mark_claimed_task_execution_started(task_id, worker_id, claimed.lease_expires_at)
    clock.value = claimed.lease_expires_at + timedelta(seconds=1)
    assert await store.reclaim_expired(query=TaskQuery(type="scan")) == []
    return claimed


async def _nothing_due(_app, task, _worker_id):
    raise AssertionError(f"nothing is claimable, but {task.id} was handled")


async def _until_idle(app, store, task_type: str, **options):
    return await asyncio.wait_for(
        run_task_worker_until_idle(
            app,
            store,
            _nothing_due,
            worker_id=f"{task_type}-worker",
            query=TaskQuery(type=task_type),
            poll_interval_s=_SLOW_POLL_S,
            **options,
        ),
        timeout=_EXIT_TIMEOUT_S,
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_until_idle_asks_for_attention_for_a_fenced_task_it_wont_settle(backend, tmp_path):
    async def run():
        clock = _Clock()
        store = _store(backend, tmp_path, clock)
        app = CayuApp(task_store=store, enable_logging=False)
        try:
            await _fenced(store, clock, "dead")

            result = await _until_idle(app, store, "scan")

            assert result.idle and result.attention_required
            assert result.next_expiry_at is None and not result.maintenance_required
            fenced = await store.load_task("dead")
            assert fenced is not None and fenced.status is TaskStatus.CLAIMED
        finally:
            await app.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_a_declared_single_worker_settles_a_fenced_task_across_until_idle_runs(backend, tmp_path):
    async def run():
        clock = _Clock()
        store = _store(backend, tmp_path, clock)
        app = CayuApp(task_store=store, enable_logging=False)
        try:
            dead = await _fenced(store, clock, "dead")
            assert dead.lease_expires_at is not None
            settle_at = dead.lease_expires_at + timedelta(seconds=120)

            first = await _until_idle(app, store, "scan", reconcile_abandoned_after_s=120)

            assert first.next_expiry_at == settle_at
            assert not first.attention_required
            waiting = await store.load_task("dead")
            assert waiting is not None and waiting.status is TaskStatus.CLAIMED

            # The host comes back when it said; a new run counts from the lease too.
            clock.value = settle_at
            second = await _until_idle(app, store, "scan", reconcile_abandoned_after_s=120)

            settled = await store.load_task("dead")
            assert settled is not None and settled.status is TaskStatus.CANCELLED
            assert second.next_expiry_at is None and not second.attention_required
        finally:
            await app.aclose()

    asyncio.run(run())


async def _recover_with(monkeypatch, error: Exception, *, parked: bool = False):
    """Recover one expired attached owner whose recovery step raises ``error``.

    The session is still running, so its recovery raises; or, with ``parked``,
    it was already interrupted and stopping its external wait raises.
    """

    clock = _Clock()
    store = InMemoryTaskStore(clock=clock, ownership_clock=clock)
    app = CayuApp(task_store=store, enable_logging=False)
    owner = await _attached_owner(store, "owner", "worker-old:continue:wake", 30)
    assert owner.lease_expires_at is not None
    clock.value = owner.lease_expires_at + timedelta(seconds=1)
    session = SimpleNamespace(
        id=owner.session_id,
        instance_id=owner.session_instance_id,
        run_epoch=0,
        status=SessionStatus.INTERRUPTED if parked else SessionStatus.RUNNING,
    )

    async def load(_session_id):
        return session

    async def not_an_external_wait(_app, _session_id):
        return False

    async def raises(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(app.session_store, "load", load)
    monkeypatch.setattr(external_wait_tasks, "recover_before_handoff", not_an_external_wait)
    monkeypatch.setattr(app, "recover_incomplete_session", raises)
    monkeypatch.setattr(external_wait_tasks, "stop_wait_left_arming", raises)
    try:
        return await task_worker._recover_expired_interrupted_task_handoffs(
            app, store, worker_id="recovery-worker", after=None, limit=10, stop=None
        )
    finally:
        await app.aclose()


def test_handoff_recovery_tells_a_person_needed_from_a_retryable_failure(monkeypatch):
    manual = asyncio.run(
        _recover_with(monkeypatch, ModelCompletionManualRecoveryRequired("No terminal response."))
    )
    assert manual.manual_recovery_required and manual.failed == frozenset()
    retryable = asyncio.run(_recover_with(monkeypatch, RuntimeError("database went away")))
    assert retryable.failed == {"owner"} and not retryable.manual_recovery_required


def test_a_wait_that_cannot_be_stopped_is_a_recovery_failure(monkeypatch):
    page = asyncio.run(_recover_with(monkeypatch, RuntimeError("wait store away"), parked=True))
    assert page.failed == {"owner"} and page.recovered == 0


def _page(*, failed=(), manual=False):
    return task_worker._InterruptedHandoffRecoveryPage(
        recovered=0,
        next_after=None,
        exhausted=True,
        failed=frozenset(failed),
        manual_recovery_required=manual,
    )


@pytest.mark.parametrize(
    ("passes", "maintenance_required", "attention_required"),
    [
        # Fails again on the pass the run retries at once: someone must look.
        ([_page(failed={"owner"}), _page(failed={"owner"})], False, True),
        # A failure the retried pass clears leaves nothing behind.
        ([_page(failed={"owner"}), _page()], False, False),
        # A different task failing is new: a later run may still clear it.
        ([_page(failed={"owner"}), _page(failed={"other"})], True, False),
        # A person must act, so the run doesn't retry.
        ([_page(manual=True)], False, True),
    ],
)
def test_until_idle_retries_a_failed_recovery_pass_once_before_it_reports(
    monkeypatch, passes, maintenance_required, attention_required
):
    async def run():
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        calls = []

        async def recover(_app, _store, *, worker_id, after, limit, stop):
            calls.append(after)
            return passes[min(len(calls), len(passes)) - 1]

        monkeypatch.setattr(task_worker, "_recover_expired_interrupted_task_handoffs", recover)

        result = await _until_idle(app, store, "scan")

        assert len(calls) == len(passes)
        assert result.idle
        assert result.maintenance_required is maintenance_required
        assert result.attention_required is attention_required
        await app.aclose()

    asyncio.run(run())


def test_a_fenced_scan_that_fails_asks_for_another_run(monkeypatch):
    import cayu.tasks.abandoned as abandoned

    async def fails(_self):
        raise RuntimeError("settlement failed")

    monkeypatch.setattr(abandoned.FencedTaskMonitor, "scan", fails)

    async def run():
        clock = _Clock()
        store = InMemoryTaskStore(clock=clock, ownership_clock=clock)
        app = CayuApp(task_store=store, enable_logging=False)
        await _fenced(store, clock, "dead")
        result = await _until_idle(app, store, "scan", reconcile_abandoned_after_s=0)
        assert result.idle and result.maintenance_required
        await app.aclose()

    asyncio.run(run())


def test_a_declared_single_worker_settles_on_a_store_without_a_lease_clock(caplog):
    caplog.set_level(logging.WARNING)

    class LeaselessStore(InMemoryTaskStore):
        """A custom store that predates TaskStore.lease_clock."""

        supports_lease_clock = False
        verified_work_mutations_are_cancellation_quiescent = True

    async def run():
        clock = _Clock()
        store = LeaselessStore(clock=clock, ownership_clock=clock)
        app = CayuApp(task_store=store, enable_logging=False)
        for task_id in ("dead-1", "dead-2"):
            await _fenced(store, clock, task_id)
            # Counted from this process's first sight, so 0 settles at once.
            result = await _until_idle(app, store, "scan", reconcile_abandoned_after_s=0)
            settled = await store.load_task(task_id)
            assert settled is not None and settled.status is TaskStatus.CANCELLED
            assert result.idle and result.next_expiry_at is None
        await app.aclose()

    asyncio.run(run())
    messages = [
        record.getMessage()
        for record in caplog.records
        if "supports_lease_clock" in record.getMessage()
    ]
    assert len(messages) == 1


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_until_idle_reports_no_time_for_claims_held_by_other_workers(backend, tmp_path):
    """A host waits out a crashed run's claims itself; live claims must not wake it."""

    async def run():
        clock = _Clock()
        store = _store(backend, tmp_path, clock)
        app = CayuApp(task_store=store, enable_logging=False)
        try:
            await _claim(store, "dead", "scan", "worker-old", LEASE_SECONDS)
            live = await _claim(store, "live", "scan", "worker-live", LEASE_SECONDS)
            await _attached_owner(store, "owner", "worker-old:continue:wake", LEASE_SECONDS)
            for _ in range(3):
                clock.value += timedelta(seconds=LEASE_SECONDS / 3)
                live = await store.heartbeat(
                    "live",
                    "worker-live",
                    lease_expires_at=live.lease_expires_at,
                    extend_seconds=LEASE_SECONDS,
                )
                result = await _until_idle(app, store, "scan")
                assert result.idle and result.next_expiry_at is None
                assert not result.maintenance_required and not result.attention_required
        finally:
            await app.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_store_reports_the_clock_it_decides_lease_expiry_by(backend, tmp_path, request):
    async def run():
        clock = _Clock()
        if backend == "postgres":
            store = PostgresTaskStore(
                request.getfixturevalue("postgres_dsn"), schema_mode=SchemaMode.CREATE
            )
        else:
            # Evidence timestamps follow another clock; leases follow the ownership clock.
            evidence = _Clock()
            evidence.value += timedelta(minutes=10)
            store = (
                InMemoryTaskStore(clock=evidence, ownership_clock=clock)
                if backend == "memory"
                else SQLiteTaskStore(
                    tmp_path / "tasks.sqlite", clock=evidence, ownership_clock=clock
                )
            )
        try:
            claimed = await _claim(store, "leased", "scan", "worker", LEASE_SECONDS)
            now = await store.lease_clock()
            assert store.supports_lease_clock
            if backend == "postgres":
                # Postgres decides leases by its own clock, whatever clock the store has.
                assert claimed.lease_expires_at - timedelta(seconds=LEASE_SECONDS) <= now
                assert now < claimed.lease_expires_at
            else:
                assert now == clock.value
        finally:
            if backend != "memory":
                await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("after_s", [0, 120])
def test_one_until_idle_run_reaches_fenced_tasks_behind_ones_it_cant_settle(monkeypatch, after_s):
    import cayu.tasks.abandoned as abandoned

    # A long-running worker's scan would stop after two; an until-idle run reads on.
    monkeypatch.setattr(abandoned, "_SCAN_PAGE", 2)
    monkeypatch.setattr(abandoned, "_SCAN_MAX", 2)

    async def run():
        clock = _Clock()
        store = InMemoryTaskStore(clock=clock, ownership_clock=clock)
        app = CayuApp(task_store=store, enable_logging=False)
        # Listed first: fenced under the until-idle worker's own id, which it never settles.
        for index in range(3):
            await _fenced(store, clock, f"own-{index}", worker_id="scan-worker")
        later = [await _fenced(store, clock, f"dead-{index}") for index in range(2)]

        for _ in range(2):
            result = await _until_idle(app, store, "scan", reconcile_abandoned_after_s=after_s)
            # Nothing a later run could do differently: no restart signal, so no loop.
            assert result.idle and result.attention_required
            assert not result.maintenance_required
            statuses = {(await store.load_task(task.id)).status for task in later}
            if after_s == 0:
                assert statuses == {TaskStatus.CANCELLED}
                assert result.next_expiry_at is None
            else:
                assert statuses == {TaskStatus.CLAIMED}
                assert result.next_expiry_at == later[0].lease_expires_at + timedelta(
                    seconds=after_s
                )
        await app.aclose()

    asyncio.run(run())


def test_more_fenced_tasks_than_one_run_reads_ask_for_attention(monkeypatch):
    import cayu.tasks.abandoned as abandoned

    monkeypatch.setattr(abandoned, "_SCAN_PAGE", 2)
    monkeypatch.setattr(abandoned, "_WHOLE_PASS_SCAN_MAX", 4)

    async def run():
        clock = _Clock()
        store = InMemoryTaskStore(clock=clock, ownership_clock=clock)
        app = CayuApp(task_store=store, enable_logging=False)
        task_ids = [f"dead-{index}" for index in range(5)]
        for task_id in task_ids:
            await _fenced(store, clock, task_id)

        first = await _until_idle(app, store, "scan", reconcile_abandoned_after_s=0)
        statuses = [(await store.load_task(task_id)).status for task_id in task_ids]
        assert statuses.count(TaskStatus.CANCELLED) == 4
        # Unusually many: someone should look, and the host isn't told to rerun at once.
        assert first.idle and first.attention_required and not first.maintenance_required

        second = await _until_idle(app, store, "scan", reconcile_abandoned_after_s=0)
        statuses = [(await store.load_task(task_id)).status for task_id in task_ids]
        assert statuses == [TaskStatus.CANCELLED] * 5
        assert second.idle and not second.attention_required
        await app.aclose()

    asyncio.run(run())
