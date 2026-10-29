"""Restartable tasks fenced after worker loss run again instead of being cancelled."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from cayu.applications import CayuApp
from cayu.storage.migrations import SchemaMode
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks import (
    AbandonedTaskSettlement,
    TaskRestartLimitReached,
    TaskRestartReason,
    complete_managed_task,
    reconcile_abandoned_tasks,
    restart_abandoned_task,
)
from cayu.tasks.abandoned import (
    MAX_ABANDONED_RESTARTS_ENV,
    RECONCILE_ABANDONED_AFTER_ENV,
    RESTART_ABANDONED_ENV,
    SINGLE_WORKER_VALIDATOR_ID,
    FencedTaskMonitor,
    abandoned_task_settlement,
)
from cayu.tasks.creation import TaskCreate
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.queries import TaskQuery
from cayu.tasks.records import Task, TaskClaimLost, TaskStatus
from cayu.tasks.store import TaskStore
from cayu.tasks.terminalization import TaskTerminalizationConflict
from cayu.tasks.worker import run_task_worker

LEASE_SECONDS = 30
SETTLE = {
    "live_worker_ids": ["worker-live"],
    "validator_id": "service.single-worker",
    "validator_version": "1",
    "settlement": "restart",
}


@pytest.fixture(autouse=True)
def _no_ambient_declaration(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (RECONCILE_ABANDONED_AFTER_ENV, RESTART_ABANDONED_ENV, MAX_ABANDONED_RESTARTS_ENV):
        monkeypatch.delenv(name, raising=False)


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


async def _close(store: TaskStore) -> None:
    close = getattr(store, "close", None)
    if close is not None:
        await close()


async def _lose_worker(store: TaskStore, clock: _Clock, task_id: str, worker_id: str) -> Task:
    """Claim and start the pending task, then let the worker's lease expire and fence it."""

    claimed = await store.claim_task(worker_id, TaskQuery(type="scan"), lease_seconds=LEASE_SECONDS)
    assert claimed is not None and claimed.id == task_id
    assert claimed.lease_expires_at is not None
    await store.mark_claimed_task_execution_started(task_id, worker_id, claimed.lease_expires_at)
    clock.value = claimed.lease_expires_at + timedelta(seconds=1)
    assert await store.reclaim_expired(query=TaskQuery(type="scan")) == []
    fenced = await store.load_task(task_id)
    assert fenced is not None and fenced.status_reason == "cancellation_requested"
    return fenced


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_restart_requeues_with_history_until_the_cap_then_cancels(
    backend: str, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="cayu.tasks.abandoned")

    async def scenario() -> None:
        clock = _Clock()
        store = _store(backend, tmp_path, clock)
        await store.create_task(
            TaskCreate(task_id="scan-1", type="scan", restartable=True, input={"repo": "x"})
        )
        first_loss = await _lose_worker(store, clock, "scan-1", "worker-a")

        [restarted] = await reconcile_abandoned_tasks(store, **SETTLE)
        assert restarted.id == "scan-1"
        assert restarted.status is TaskStatus.PENDING
        assert restarted.restart_count == 1
        assert restarted.input == {"repo": "x"}
        assert restarted.worker_id is None and restarted.started_at is None
        [record] = restarted.restart_history
        assert record.reason is TaskRestartReason.WORKER_LOST
        assert record.previous_worker_id == "worker-a"
        assert record.previous_lease_expires_at == first_loss.lease_expires_at
        assert record.restarted_at == clock.value
        assert record.validator_id == "service.single-worker"
        assert record.evidence_id == f"lease-expired-at:{first_loss.lease_expires_at.isoformat()}"
        assert restarted.status_payload == {
            "restart": {
                "count": 1,
                "previous_session_id": None,
                "reason": "worker_lost",
                "previous_worker_id": "worker-a",
                "previous_lease_expires_at": first_loss.lease_expires_at.isoformat(),
                "validator_id": "service.single-worker",
                "validator_version": "1",
                "evidence_id": record.evidence_id,
            }
        }
        if backend == "sqlite":
            await _close(store)
            store = _store(backend, tmp_path, clock)
            assert await store.load_task("scan-1") == restarted

        # The history survives the next claim and fence, which replace status_payload.
        await _lose_worker(store, clock, "scan-1", "worker-b")
        [second] = await reconcile_abandoned_tasks(store, **SETTLE)
        assert second.status is TaskStatus.PENDING and second.restart_count == 2
        assert [r.previous_worker_id for r in second.restart_history] == ["worker-a", "worker-b"]

        # The default cap is two restarts: the third loss cancels, saying why.
        third_loss = await _lose_worker(store, clock, "scan-1", "worker-c")
        [cancelled] = await reconcile_abandoned_tasks(store, **SETTLE)
        assert cancelled.status is TaskStatus.CANCELLED
        assert cancelled.restart_count == 2
        assert [r.previous_worker_id for r in cancelled.restart_history] == [
            "worker-a",
            "worker-b",
        ]
        evidence = cancelled.status_payload["cancellation_reconciliation"]["evidence"]  # type: ignore[index]
        assert evidence["evidence_id"] == (
            f"restart-limit-reached:2:lease-expired-at:{third_loss.lease_expires_at.isoformat()}"
        )
        await _close(store)

    asyncio.run(scenario())
    [message] = [
        record.getMessage() for record in caplog.records if "its limit" in record.getMessage()
    ]
    assert "Task scan-1 (type scan) was already restarted 2 time(s)" in message


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_non_restartable_tasks_and_cancel_settlement_keep_cancelling(
    backend: str, tmp_path: Path
) -> None:
    async def scenario() -> None:
        clock = _Clock()
        store = _store(backend, tmp_path, clock)
        await store.create_task(TaskCreate(task_id="plain", type="scan"))
        plain = await _lose_worker(store, clock, "plain", "worker-a")
        assert "wasn't created with restartable=True" in abandoned_task_settlement(plain)
        [cancelled] = await reconcile_abandoned_tasks(store, **SETTLE)
        assert cancelled.id == "plain" and cancelled.status is TaskStatus.CANCELLED
        assert cancelled.restart_count == 0 and cancelled.restart_history == ()
        evidence = cancelled.status_payload["cancellation_reconciliation"]["evidence"]  # type: ignore[index]
        assert evidence["evidence_id"].startswith("lease-expired-at:")

        # The default settlement cancels a restartable task too.
        await store.create_task(TaskCreate(task_id="restartable", type="scan", restartable=True))
        restartable = await _lose_worker(store, clock, "restartable", "worker-b")
        assert "restart_abandoned_task" in abandoned_task_settlement(restartable)
        [default] = await reconcile_abandoned_tasks(
            store, **{**SETTLE, "settlement": AbandonedTaskSettlement.CANCEL}
        )
        assert default.status is TaskStatus.CANCELLED and default.restart_count == 0

        with pytest.raises(ValueError, match="'cancel' or 'restart'"):
            await reconcile_abandoned_tasks(store, **{**SETTLE, "settlement": "retry"})
        with pytest.raises(ValueError, match="max_restarts"):
            await reconcile_abandoned_tasks(store, **SETTLE, max_restarts=-1)
        await _close(store)

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_the_tasks_own_cap_wins_and_the_standalone_restart_checks_its_claim(
    backend: str, tmp_path: Path
) -> None:
    async def scenario() -> None:
        clock = _Clock()
        store = _store(backend, tmp_path, clock)
        restart = {
            "validator_id": "service.restart",
            "validator_version": "1",
            "evidence_id": "process-exit:41",
        }
        await store.create_task(
            TaskCreate(task_id="once", type="scan", restartable=True, max_restarts=1)
        )
        await _lose_worker(store, clock, "once", "worker-a")
        restarted = await restart_abandoned_task(store, "once", **restart, max_restarts=5)
        assert restarted.status is TaskStatus.PENDING and restarted.max_restarts == 1
        # Repeating the same restart returns the task unchanged.
        assert await restart_abandoned_task(store, "once", **restart) == restarted

        # Its own cap of one restart wins over the caller's five.
        fenced = await _lose_worker(store, clock, "once", "worker-b")
        with pytest.raises(TaskRestartLimitReached) as limit:
            await restart_abandoned_task(store, "once", **{**restart, "evidence_id": "exit:42"})
        assert (limit.value.restarts, limit.value.max_restarts) == (1, 1)
        assert await store.load_task("once") == fenced

        # A live claim, and a task that isn't restartable, are refused.
        await store.create_task(TaskCreate(task_id="live", type="scan", restartable=True))
        live = await store.claim_task("worker-c", TaskQuery(type="scan"), lease_seconds=60)
        assert live is not None and live.id == "live"
        with pytest.raises(TaskTerminalizationConflict, match="fenced after its worker"):
            await restart_abandoned_task(store, "live", **restart)
        await store.create_task(TaskCreate(task_id="plain", type="scan"))
        await _lose_worker(store, clock, "plain", "worker-d")
        with pytest.raises(TaskTerminalizationConflict, match="restartable=True"):
            await restart_abandoned_task(store, "plain", **restart)
        await _close(store)

    asyncio.run(scenario())


def test_the_store_refuses_a_lease_it_has_not_seen_expire() -> None:
    from cayu.tasks.restart import AbandonedTaskRestartRequest

    async def scenario() -> None:
        clock = _Clock()
        store = InMemoryTaskStore(ownership_clock=clock)
        await store.create_task(TaskCreate(task_id="job", type="scan", restartable=True))
        fenced = await _lose_worker(store, clock, "job", "worker-a")
        assert fenced.lease_expires_at is not None
        request = AbandonedTaskRestartRequest(
            task_id="job",
            original_worker_id="worker-a",
            original_lease_expires_at=fenced.lease_expires_at,
            validator_id="v",
            validator_version="1",
            evidence_id="e",
        )
        clock.value = fenced.lease_expires_at - timedelta(seconds=1)
        with pytest.raises(TaskClaimLost, match="has not expired"):
            await store.restart_abandoned_task(request)
        clock.value = fenced.lease_expires_at + timedelta(seconds=1)
        with pytest.raises(TaskClaimLost, match="no longer held"):
            await store.restart_abandoned_task(
                request.model_copy(update={"original_worker_id": "worker-b"})
            )
        assert (await store.restart_abandoned_task(request)).status is TaskStatus.PENDING

    asyncio.run(scenario())


def test_restart_settings_need_a_restartable_task() -> None:
    with pytest.raises(ValidationError, match="max_restarts requires restartable"):
        TaskCreate(type="scan", max_restarts=1)
    with pytest.raises(ValidationError):
        TaskCreate(type="scan", restartable=True, max_restarts=101)
    task = asyncio.run(InMemoryTaskStore().create_task(TaskCreate(type="scan", restartable=True)))
    # Unset restart settings stay out of serialized tasks, so older readers accept them.
    dumped = task.model_dump(mode="json")
    assert "max_restarts" not in dumped and "restart_history" not in dumped
    assert "max_restarts" not in TaskCreate(type="scan").model_dump(mode="json")
    with pytest.raises(ValidationError, match="max_restarts requires restartable"):
        task.model_validate({**dumped, "restartable": False, "max_restarts": 1})


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_single_worker_restarts_a_lost_task_and_the_handler_sees_it(
    backend: str, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="cayu.tasks.abandoned")

    async def scenario() -> None:
        clock = _Clock()
        store = _store(backend, tmp_path, clock)
        app = CayuApp(task_store=store, enable_logging=False)
        await store.create_task(TaskCreate(task_id="scan-1", type="scan", restartable=True))
        await _lose_worker(store, clock, "scan-1", "worker-old")
        # The new worker's claims need a live lease by the store's clock.
        clock.value = datetime.now(UTC)
        seen: list[Task] = []
        stop = asyncio.Event()

        async def handler(_app: CayuApp, task: Task, worker_id: str) -> None:
            seen.append(task)
            await complete_managed_task(store, task, worker_id, {"ok": True})
            stop.set()

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
                    reconcile_abandoned_after_s=0,
                    restart_abandoned=True,
                    stop=stop,
                ),
                timeout=10,
            )
        finally:
            await app.aclose()
        [task] = seen
        assert task.restart_count == 1
        assert task.status_payload is not None
        assert task.status_payload["restart"]["previous_worker_id"] == "worker-old"
        assert task.status_payload["restart"]["validator_id"] == SINGLE_WORKER_VALIDATOR_ID
        done = await store.load_task("scan-1")
        assert done is not None and done.status is TaskStatus.COMPLETED
        assert len(done.restart_history) == 1
        await _close(store)

    asyncio.run(scenario())
    [message] = [r.getMessage() for r in caplog.records if "Restarted task" in r.getMessage()]
    assert "Restarted task scan-1 (type scan)" in message
    assert "restart 1 of 2 after worker loss" in message


def test_the_monitor_says_what_it_will_do_with_each_fenced_task(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="cayu.tasks.abandoned")

    async def scenario() -> None:
        clock = _Clock()
        store = InMemoryTaskStore(ownership_clock=clock)
        await store.create_task(TaskCreate(task_id="plain", type="scan"))
        await _lose_worker(store, clock, "plain", "worker-old")
        await store.create_task(
            TaskCreate(task_id="capped", type="scan", restartable=True, max_restarts=0)
        )
        await _lose_worker(store, clock, "capped", "worker-old")
        await store.create_task(TaskCreate(task_id="restartable", type="scan", restartable=True))
        await _lose_worker(store, clock, "restartable", "worker-old")
        monitor = FencedTaskMonitor(
            store,
            worker_id="worker-new",
            query=TaskQuery(type="scan"),
            reconcile_abandoned_after_s=60,
            clock=lambda: 0.0,
            settlement=AbandonedTaskSettlement.RESTART,
        )
        assert await monitor.scan() == []

    asyncio.run(scenario())
    messages = {
        record.getMessage().split()[1]: record.getMessage()
        for record in caplog.records
        if "is fenced" in record.getMessage()
    }
    assert "can't be restarted" in messages["plain"]
    assert "already restarted 0 time(s) after worker loss, its limit" in messages["capped"]
    assert "will restart it once it has stayed fenced for 60 seconds" in messages["restartable"]
    assert "(restart 1 of 2 after worker loss)" in messages["restartable"]


def test_worker_restart_options_are_validated(monkeypatch: pytest.MonkeyPatch) -> None:
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

    with pytest.raises(ValueError, match="needs reconcile_abandoned_after_s"):
        asyncio.run(run(restart_abandoned=True))
    with pytest.raises(ValueError, match="only applies with restart_abandoned"):
        asyncio.run(run(reconcile_abandoned_after_s=0, max_abandoned_restarts=3))
    with pytest.raises(ValueError, match="max_abandoned_restarts"):
        asyncio.run(
            run(reconcile_abandoned_after_s=0, restart_abandoned=True, max_abandoned_restarts=-1)
        )
    assert asyncio.run(run(reconcile_abandoned_after_s=0, restart_abandoned=True)) == 0
    monkeypatch.setenv(RESTART_ABANDONED_ENV, "maybe")
    with pytest.raises(ValueError, match=RESTART_ABANDONED_ENV):
        asyncio.run(run(reconcile_abandoned_after_s=0))
    monkeypatch.setenv(RESTART_ABANDONED_ENV, "1")
    monkeypatch.setenv(MAX_ABANDONED_RESTARTS_ENV, "many")
    with pytest.raises(ValueError, match=MAX_ABANDONED_RESTARTS_ENV):
        asyncio.run(run(reconcile_abandoned_after_s=0))
    monkeypatch.setenv(MAX_ABANDONED_RESTARTS_ENV, "4")
    assert asyncio.run(run(reconcile_abandoned_after_s=0)) == 0


def test_sqlite_revision_135_adds_restart_columns_to_existing_tasks(tmp_path: Path) -> None:
    async def scenario() -> None:
        path = tmp_path / "upgrade.db"
        original = SQLiteTaskStore(path)
        created = await original.create_task(TaskCreate(type="existing", restartable=True))
        await original.close()
        with sqlite3.connect(path) as connection:
            connection.execute("ALTER TABLE cayu_tasks DROP COLUMN max_restarts")
            connection.execute("ALTER TABLE cayu_tasks DROP COLUMN restart_history_json")
            connection.execute("DELETE FROM cayu_schema_migrations WHERE revision > 134")
            connection.execute("PRAGMA user_version = 134")
        upgraded = SQLiteTaskStore(path, schema_mode=SchemaMode.MIGRATE)
        try:
            loaded = await upgraded.load_task(created.id)
            assert loaded == created
            assert loaded is not None
            assert loaded.max_restarts is None and loaded.restart_history == ()
            new = await upgraded.create_task(
                TaskCreate(type="new", restartable=True, max_restarts=3)
            )
            assert (await upgraded.load_task(new.id)).max_restarts == 3  # type: ignore[union-attr]
        finally:
            await upgraded.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("error", [None, {"code": {"reason": "stop"}}, {"code": ["a", "b"]}])
def test_an_explicit_cancellation_is_not_undone_by_a_restart(
    backend: str, tmp_path: Path, error: dict | None
) -> None:
    async def scenario() -> None:
        clock = _Clock()
        store = _store(backend, tmp_path, clock)
        await store.create_task(TaskCreate(task_id="job", type="scan", restartable=True))
        claimed = await store.claim_task("worker-a", TaskQuery(type="scan"), lease_seconds=30)
        assert claimed is not None and claimed.lease_expires_at is not None
        await store.mark_claimed_task_execution_started("job", "worker-a", claimed.lease_expires_at)
        # A person cancels the running task, then its worker dies before draining it.
        await store.cancel_task("job", error)
        clock.value = claimed.lease_expires_at + timedelta(seconds=1)
        with pytest.raises(TaskTerminalizationConflict, match="requested explicitly"):
            await restart_abandoned_task(
                store, "job", validator_id="v", validator_version="1", evidence_id="e"
            )
        [settled] = await reconcile_abandoned_tasks(store, **SETTLE)
        assert settled.status is TaskStatus.CANCELLED and settled.restart_count == 0
        await _close(store)

    asyncio.run(scenario())


def test_replaying_restart_evidence_never_restarts_a_later_claim() -> None:
    from cayu.tasks.restart import AbandonedTaskRestartRequest

    async def scenario() -> None:
        clock = _Clock()
        store = InMemoryTaskStore(ownership_clock=clock)
        await store.create_task(TaskCreate(task_id="job", type="scan", restartable=True))
        evidence = {"validator_id": "platform", "validator_version": "1", "evidence_id": "a-exited"}
        first = await _lose_worker(store, clock, "job", "worker-a")
        restarted = await restart_abandoned_task(store, "job", **evidence)
        assert restarted.restart_count == 1
        # Worker B claims it and is lost too; A's evidence must not restart B's claim.
        fenced_b = await _lose_worker(store, clock, "job", "worker-b")
        assert await restart_abandoned_task(store, "job", **evidence) == fenced_b
        assert fenced_b.lease_expires_at is not None and first.lease_expires_at is not None
        with pytest.raises(TaskTerminalizationConflict, match="different worker claim"):
            await store.restart_abandoned_task(
                AbandonedTaskRestartRequest(
                    task_id="job",
                    original_worker_id="worker-b",
                    original_lease_expires_at=fenced_b.lease_expires_at,
                    **evidence,
                )
            )
        # The store recognizes the exact first restart and changes nothing.
        replay = AbandonedTaskRestartRequest(
            task_id="job",
            original_worker_id="worker-a",
            original_lease_expires_at=first.lease_expires_at,
            **evidence,
        )
        assert await store.restart_abandoned_task(replay) == fenced_b

    asyncio.run(scenario())


def test_a_batch_restart_refuses_a_claim_that_changed_after_listing() -> None:
    from cayu.tasks.abandoned import AbandonedTaskSettlement, _settle_abandoned

    async def scenario() -> None:
        clock = _Clock()
        store = InMemoryTaskStore(ownership_clock=clock)
        await store.create_task(TaskCreate(task_id="job", type="scan", restartable=True))
        listed = await _lose_worker(store, clock, "job", "worker-a")
        # Someone else restarts it, and a live worker claims and is fenced again.
        await restart_abandoned_task(
            store, "job", validator_id="other", validator_version="1", evidence_id="x"
        )
        await _lose_worker(store, clock, "job", "worker-live")
        settled = await _settle_abandoned(
            store,
            [listed],
            validator_id="v",
            validator_version="1",
            reconciled_by=None,
            settlement=AbandonedTaskSettlement.RESTART,
            max_restarts=2,
        )
        assert settled == []
        current = await store.load_task("job")
        assert current is not None and current.worker_id == "worker-live"

    asyncio.run(scenario())


def test_a_capped_batch_never_cancels_a_replacement_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.tasks.abandoned as abandoned

    async def scenario() -> None:
        clock = _Clock()
        store = InMemoryTaskStore(ownership_clock=clock)
        await store.create_task(TaskCreate(task_id="job", type="scan", restartable=True))
        listed = await _lose_worker(store, clock, "job", "worker-a")

        async def limit_reached_then_raced(*_args: object, **_kwargs: object) -> Task:
            # While this batch finds its cap reached, another reconciler with a
            # higher cap restarts the task, and its replacement is fenced in turn.
            await restart_abandoned_task(
                store, "job", validator_id="other", validator_version="1", evidence_id="x"
            )
            await _lose_worker(store, clock, "job", "worker-live")
            raise TaskRestartLimitReached("job", restarts=0, max_restarts=0)

        monkeypatch.setattr(abandoned, "restart_observed_abandoned_task", limit_reached_then_raced)
        settled = await abandoned._settle_abandoned(
            store,
            [listed],
            validator_id="v",
            validator_version="1",
            reconciled_by=None,
            settlement=AbandonedTaskSettlement.RESTART,
            max_restarts=0,
        )
        # The cancellation is bound to the listed claim, so it skips the replacement.
        assert settled == []
        current = await store.load_task("job")
        assert current is not None
        assert current.status is TaskStatus.CLAIMED and current.worker_id == "worker-live"

    asyncio.run(scenario())
