from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError
from tests.core.task_invocation_fixtures import task_backed_session_invocation
from tests.core.task_terminalization_conformance import ordinary_cancellation_reconciliation_request

from cayu import (
    CayuApp,
    InMemoryTaskStore,
    TaskCreate,
    TaskQuery,
    TaskRetryAttemptDisposition,
    TaskRetryPolicy,
    TaskRetrySettlementRequest,
    TaskStatus,
    TaskTerminalizationRequest,
    TaskTerminalKind,
)
from cayu.storage.migrations import SchemaMode
from cayu.storage.postgres import PostgresTaskStore
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks.scheduling import (
    TaskMisfirePolicy,
    TaskRescheduleRequest,
    TaskScheduleCancelRequest,
    TaskScheduleConflict,
    TaskScheduleEligibility,
    TaskScheduleEventType,
    TaskSchedulePolicy,
    copy_task_schedule_policy,
    task_schedule_eligibility,
)
from cayu.tasks.worker import complete_managed_task, run_task_worker


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("deadline", ["expiry", "skip"])
def test_pending_schedule_retry_deadline_precedes_schedule_nonexecution(
    backend, deadline, tmp_path, request
):
    dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None

    async def run():
        clock = [datetime.now(UTC)]
        path = tmp_path / "overlap.sqlite"

        def open_store():
            if backend == "memory":
                return InMemoryTaskStore(clock=lambda: clock[0], ownership_clock=lambda: clock[0])
            if backend == "sqlite":
                return SQLiteTaskStore(
                    path, clock=lambda: clock[0], ownership_clock=lambda: clock[0]
                )
            return PostgresTaskStore(dsn, schema_mode=SchemaMode.CREATE, clock=lambda: clock[0])

        store = open_store()
        creation = TaskCreate(
            task_id=f"overlap-{deadline}",
            type="followup",
            available_at=clock[0] + timedelta(seconds=1),
            schedule_policy=TaskSchedulePolicy(
                expires_at=clock[0] + timedelta(seconds=2) if deadline == "expiry" else None,
                misfire_policy=TaskMisfirePolicy.SKIP
                if deadline == "skip"
                else TaskMisfirePolicy.FIRE_ONCE,
                misfire_grace_seconds=0,
            ),
            retry_policy=TaskRetryPolicy(max_attempts=2, max_elapsed_seconds=3),
        )
        try:
            created = await store.create_task(creation)
            clock[0] += timedelta(seconds=4)
            assert await store.claim_task("worker") is None
            terminal = await store.load_task(created.id)
            assert terminal.status is TaskStatus.FAILED
            assert terminal.schedule == created.schedule
            assert terminal.retry_series.disposition == "elapsed_exhausted"
            assert terminal.retry_series.cumulative_tokens == 0
            key = terminal.status_payload["settlement_idempotency_key"]
            receipt = await store.load_task_retry_settlement(created.id, key)
            assert receipt.task == terminal
            assert receipt.successor is None
            history = await store.list_task_schedule_events(created.id)
            assert [event.type for event in history] == [
                TaskScheduleEventType.SCHEDULED,
                TaskScheduleEventType.FAILED,
            ]
            if backend != "memory":
                await store.close()
                store = open_store()
            assert await store.create_task(creation) == terminal
            assert await store.claim_task("replacement") is None
            assert await store.load_task(created.id) == terminal
            assert await store.load_task_retry_settlement(created.id, key) == receipt
            assert await store.list_task_schedule_events(created.id) == history
        finally:
            if backend != "memory":
                await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("microsecond", [0, 123456, 999999])
def test_sqlite_held_schedule_expiry_exact_instant(tmp_path, microsecond):
    async def run():
        clock = [datetime(2026, 9, 14, tzinfo=UTC)]
        expiry = (clock[0] + timedelta(seconds=2)).replace(microsecond=microsecond)
        store = SQLiteTaskStore(tmp_path / "expiry.sqlite", clock=lambda: clock[0])
        try:
            await store.create_task(
                TaskCreate(
                    task_id="held",
                    type="followup",
                    available_at=clock[0] + timedelta(seconds=1),
                    schedule_policy=TaskSchedulePolicy(expires_at=expiry),
                )
            )
            await store.pause_task("held")
            clock[0] = expiry - timedelta(microseconds=1)
            wake = await store.next_task_schedule_wakeup()
            assert wake.next_expiry_at == expiry
            assert not wake.maintenance_required
            assert await store.claim_task("before") is None
            assert (await store.load_task("held")).status is TaskStatus.PAUSED
            clock[0] = expiry
            wake = await store.next_task_schedule_wakeup()
            assert wake.next_expiry_at is None
            assert wake.maintenance_required
            assert await store.claim_task("at") is None
            terminal = await store.load_task("held")
            assert terminal.status is TaskStatus.CANCELLED
            assert terminal.schedule.revision == 2
            history = await store.list_task_schedule_events("held")
            assert history[-1].type is TaskScheduleEventType.EXPIRED
            clock[0] += timedelta(microseconds=1)
            assert await store.claim_task("after") is None
            assert await store.load_task("held") == terminal
            assert await store.list_task_schedule_events("held") == history
            assert not (await store.next_task_schedule_wakeup()).maintenance_required
        finally:
            await store.close()

    asyncio.run(run())


def test_sqlite_schedule_wakeup_orders_whole_and_fractional_expiries(tmp_path):
    async def run():
        clock = [datetime(2026, 9, 14, tzinfo=UTC)]
        first_expiry = clock[0] + timedelta(seconds=2)
        store = SQLiteTaskStore(tmp_path / "ordered-expiry.sqlite", clock=lambda: clock[0])
        try:
            for fraction in [123456, 0, 999999]:
                await store.create_task(
                    TaskCreate(
                        task_id=f"expiry-{fraction}",
                        type="followup",
                        available_at=clock[0] + timedelta(seconds=1),
                        schedule_policy=TaskSchedulePolicy(
                            expires_at=first_expiry.replace(microsecond=fraction)
                        ),
                    )
                )
                await store.pause_task(f"expiry-{fraction}")
            assert (await store.next_task_schedule_wakeup()).next_expiry_at == first_expiry
            clock[0] = first_expiry
            assert await store.claim_task("worker") is None
            wake = await store.next_task_schedule_wakeup()
            assert wake.next_expiry_at == first_expiry.replace(microsecond=123456)
            assert not wake.maintenance_required
            assert (await store.load_task("expiry-0")).status is TaskStatus.CANCELLED
            assert (await store.load_task("expiry-123456")).status is TaskStatus.PAUSED
            assert (await store.load_task("expiry-999999")).status is TaskStatus.PAUSED
        finally:
            await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("policy", list(TaskMisfirePolicy))
@pytest.mark.parametrize("offset", [-1, 0, 60, 61, 120])
def test_task_schedule_store_clock_boundaries(policy, offset):
    due = datetime(2026, 9, 14, tzinfo=UTC)
    settings = TaskSchedulePolicy(expires_at=due + timedelta(seconds=120), misfire_policy=policy)
    result = task_schedule_eligibility(
        available_at=due, policy=settings, as_of=due + timedelta(seconds=offset)
    )
    if offset < 0:
        expected = TaskScheduleEligibility.FUTURE
    elif offset == 120:
        expected = TaskScheduleEligibility.EXPIRED
    elif offset > 60:
        expected = (
            TaskScheduleEligibility.SKIPPED
            if policy is TaskMisfirePolicy.SKIP
            else TaskScheduleEligibility.MISFIRED
        )
    else:
        expected = TaskScheduleEligibility.ELIGIBLE
    assert result is expected


def test_task_schedule_policy_rejects_invalid_window_and_boolean_revision():
    due = datetime(2026, 9, 14, tzinfo=UTC)
    with pytest.raises(ValidationError, match="expiry must be later"):
        TaskRescheduleRequest(
            task_id="followup",
            operation_id="reschedule",
            expected_revision=1,
            available_at=due,
            policy=TaskSchedulePolicy(expires_at=due),
        )
    with pytest.raises(ValidationError):
        TaskRescheduleRequest(
            task_id="followup",
            operation_id="reschedule",
            expected_revision=True,
            available_at=due,
        )
    with pytest.raises(ValidationError, match="timezone-aware"):
        TaskSchedulePolicy(expires_at=due.replace(tzinfo=None))


@pytest.mark.parametrize("retry", [False, True])
@pytest.mark.parametrize("publication_failure", [False, True])
def test_postgres_schedule_cancellation_is_atomic_and_replayable(
    postgres_dsn, retry, publication_failure
):
    fail_publication = [publication_failure]

    class SchedulingStorage(PostgresTaskStore):
        async def _record_schedule_transition(self, cur, prior, current, *, operation_id=None):
            await super()._record_schedule_transition(
                cur, prior, current, operation_id=operation_id
            )
            if prior is not None and fail_publication[0]:
                fail_publication[0] = False
                raise RuntimeError("cancel journal publication failed")

    async def run():
        task_id = f"scheduled-cancel-{retry}-{publication_failure}"
        store = SchedulingStorage(postgres_dsn, schema_mode=SchemaMode.CREATE)
        request = TaskScheduleCancelRequest(
            task_id=task_id, operation_id="cancel", expected_revision=1
        )
        try:
            created = await store.create_task(
                TaskCreate(
                    task_id=task_id,
                    type="followup",
                    available_at=datetime.now(UTC) + timedelta(hours=1),
                    schedule_policy=TaskSchedulePolicy(),
                    retry_policy=TaskRetryPolicy(max_attempts=2) if retry else None,
                )
            )
            with pytest.raises(TaskScheduleConflict):
                await store.cancel_task(task_id)
            if publication_failure:
                with pytest.raises(RuntimeError, match="journal publication"):
                    await store.cancel_scheduled_task(request)
                assert await store.load_task(task_id) == created
                assert len(await store.list_task_schedule_events(task_id)) == 1
            receipt = await store.cancel_scheduled_task(request)
            terminal = await store.load_task(task_id)
            assert terminal is not None and terminal.status is TaskStatus.CANCELLED
            assert terminal.schedule == receipt.schedule
            if retry:
                assert terminal.retry_series is not None
                assert terminal.retry_series.disposition == "cancelled"
                assert terminal.status_payload is not None
                settlement = await store.load_task_retry_settlement(
                    task_id, terminal.status_payload["settlement_idempotency_key"]
                )
                assert settlement.task == terminal
        finally:
            await store.close()
        reopened = SchedulingStorage(postgres_dsn, schema_mode=SchemaMode.VALIDATE)
        try:
            assert await reopened.cancel_scheduled_task(request) == receipt
            assert await reopened.load_task(task_id) == terminal
            assert [event.type for event in await reopened.list_task_schedule_events(task_id)] == [
                TaskScheduleEventType.SCHEDULED,
                TaskScheduleEventType.CANCELLED,
            ]
        finally:
            await reopened.close()

    asyncio.run(run())


@pytest.mark.parametrize("case", ["on_time", "late", "skip", "expired"])
@pytest.mark.parametrize("retry", [False, True])
def test_postgres_schedule_claim_rechecks_policy_and_records_one_admission(
    postgres_dsn, case, retry
):
    SchedulingStorage = PostgresTaskStore

    async def run():
        clock = [datetime.now(UTC)]
        due = clock[0] + timedelta(seconds=10)
        identity = f"schedule-claim-{case}-{retry}"
        store = SchedulingStorage(
            postgres_dsn, schema_mode=SchemaMode.CREATE, clock=lambda: clock[0]
        )
        other = SchedulingStorage(
            postgres_dsn, schema_mode=SchemaMode.VALIDATE, clock=lambda: clock[0]
        )
        query = TaskQuery(type=identity)
        try:
            await store.create_task(
                TaskCreate(
                    task_id=identity,
                    type=identity,
                    available_at=due,
                    schedule_policy=TaskSchedulePolicy(
                        misfire_policy=TaskMisfirePolicy.SKIP
                        if case == "skip"
                        else TaskMisfirePolicy.FIRE_ONCE,
                        expires_at=due + timedelta(seconds=120),
                    ),
                    retry_policy=TaskRetryPolicy(max_attempts=2) if retry else None,
                )
            )
            assert await store.claim_task("early", query) is None
            wakeup = await other.next_task_schedule_wakeup(query)
            assert wakeup.as_of == clock[0]
            assert wakeup.next_available_at == due
            assert wakeup.next_expiry_at == due + timedelta(seconds=120)
            assert wakeup.maintenance_required is False
            clock[0] = due + timedelta(
                seconds=0 if case == "on_time" else 120 if case == "expired" else 61
            )
            wakeup = await other.next_task_schedule_wakeup(query)
            assert wakeup.next_available_at is None
            assert wakeup.maintenance_required is (case in {"skip", "expired"})
            results = await asyncio.gather(
                store.claim_task("first", query), other.claim_task("second", query)
            )
            admitted = [task for task in results if task is not None]
            current = await store.load_task(identity)
            assert current is not None and current.schedule is not None
            wakeup = await other.next_task_schedule_wakeup(query)
            assert wakeup.next_available_at is None
            assert wakeup.next_expiry_at is None
            assert wakeup.maintenance_required is False
            if case in {"skip", "expired"}:
                assert admitted == []
                assert current.status is TaskStatus.CANCELLED
                assert current.schedule.admitted_at is None
                expected = (
                    TaskScheduleEventType.SKIPPED
                    if case == "skip"
                    else TaskScheduleEventType.EXPIRED
                )
                assert [
                    event.type for event in await store.list_task_schedule_events(identity)
                ] == [
                    TaskScheduleEventType.SCHEDULED,
                    expected,
                ]
                if retry:
                    assert (
                        current.retry_series is not None
                        and current.retry_series.disposition == "cancelled"
                    )
            else:
                assert len(admitted) == 1
                assert current.schedule.admitted_at == clock[0]
                assert current.schedule.revision == 2
                assert [
                    event.type for event in await store.list_task_schedule_events(identity)
                ] == [
                    TaskScheduleEventType.SCHEDULED,
                    TaskScheduleEventType.ELIGIBLE
                    if case == "on_time"
                    else TaskScheduleEventType.MISFIRED,
                    TaskScheduleEventType.CLAIMED,
                ]
                await store.cancel_scheduled_task(
                    TaskScheduleCancelRequest(
                        task_id=identity,
                        operation_id="cancel-live",
                        expected_revision=2,
                    )
                )
                requested = await store.load_task(identity)
                assert requested is not None and requested.status is TaskStatus.CLAIMED
                assert requested.worker_id == current.worker_id
                assert (await store.list_task_schedule_events(identity))[
                    -1
                ].type is TaskScheduleEventType.CANCELLATION_REQUESTED
        finally:
            await other.close()
            await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("misfire", list(TaskMisfirePolicy))
@pytest.mark.parametrize("hold_method", ["pause_task", "block_task", "mark_task_needs_attention"])
def test_postgres_held_schedule_expires_without_dispatch(postgres_dsn, misfire, hold_method):
    SchedulingStorage = PostgresTaskStore

    async def run():
        clock = [datetime.now(UTC)]
        identity = f"held-{misfire}-{hold_method}"
        due = clock[0] + timedelta(seconds=1)
        query = TaskQuery(type=identity)
        store = SchedulingStorage(
            postgres_dsn, schema_mode=SchemaMode.CREATE, clock=lambda: clock[0]
        )
        try:
            await store.create_task(
                TaskCreate(
                    task_id=identity,
                    type=identity,
                    available_at=due,
                    schedule_policy=TaskSchedulePolicy(
                        misfire_policy=misfire,
                        misfire_grace_seconds=0,
                        expires_at=due + timedelta(seconds=1)
                        if misfire is TaskMisfirePolicy.FIRE_ONCE
                        else None,
                    ),
                    retry_policy=TaskRetryPolicy(max_attempts=2),
                )
            )
            held = await getattr(store, hold_method)(identity)
            clock[0] = due + timedelta(seconds=2)
            assert (await store.next_task_schedule_wakeup(query)).maintenance_required
            assert await store.claim_task("unrelated", TaskQuery(type="unrelated-held")) is None
            assert await store.load_task(identity) == held
        finally:
            await store.close()
        reopened = SchedulingStorage(
            postgres_dsn, schema_mode=SchemaMode.VALIDATE, clock=lambda: clock[0]
        )
        try:
            assert await reopened.claim_task("worker", query) is None
            task = await reopened.load_task(identity)
            assert task is not None and task.status is TaskStatus.CANCELLED
            assert task.retry_series is not None and task.retry_series.disposition == "cancelled"
            assert task.status_payload is not None
            receipt = await reopened.load_task_retry_settlement(
                identity, task.status_payload["settlement_idempotency_key"]
            )
            assert receipt is not None and receipt.task == task
            assert not (await reopened.next_task_schedule_wakeup(query)).maintenance_required
            assert [event.type for event in await reopened.list_task_schedule_events(identity)] == [
                TaskScheduleEventType.SCHEDULED,
                TaskScheduleEventType.HELD,
                TaskScheduleEventType.EXPIRED
                if misfire is TaskMisfirePolicy.FIRE_ONCE
                else TaskScheduleEventType.SKIPPED,
            ]
            assert await reopened.claim_task("again", query) is None
            assert len(await reopened.list_task_schedule_events(identity)) == 3
        finally:
            await reopened.close()

    asyncio.run(run())


@pytest.mark.parametrize("attach", [False, True])
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("receipt", [False, True])
def test_postgres_schedule_lifecycle_evidence_survives_reopen(postgres_dsn, attach, fail, receipt):
    SchedulingStorage = PostgresTaskStore

    async def run():
        identity = f"schedule-lifecycle-{attach}-{fail}-{receipt}"
        store = SchedulingStorage(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            await store.create_task(
                TaskCreate(
                    task_id=identity,
                    type=identity,
                    available_at=datetime.now(UTC),
                    schedule_policy=TaskSchedulePolicy(),
                )
            )
            await store.pause_task(identity)
            await store.resume_task(identity)
            claimed = await store.claim_task("first", TaskQuery(type=identity))
            assert claimed is not None and claimed.lease_expires_at is not None
            await store.release_task(identity, "first", lease_expires_at=claimed.lease_expires_at)
            second = await store.claim_task("second", TaskQuery(type=identity))
            assert second is not None and second.lease_expires_at is not None
            assert second.schedule == claimed.schedule
            if attach:
                await store.attach_task(
                    identity,
                    session_id=f"session-{identity}",
                    worker_id="second",
                    lease_expires_at=second.lease_expires_at,
                    session_invocation=await task_backed_session_invocation(
                        store, identity, f"session-{identity}"
                    ),
                )
            else:
                started = await store.mark_claimed_task_execution_started(
                    identity, "second", second.lease_expires_at
                )
                assert (
                    await store.mark_claimed_task_execution_started(
                        identity, "second", second.lease_expires_at
                    )
                    == started
                )
            if receipt:
                terminal_request = TaskTerminalizationRequest(
                    task_id=identity,
                    worker_id="second",
                    lease_expires_at=second.lease_expires_at,
                    idempotency_key="terminal",
                    kind=TaskTerminalKind.FAILED if fail else TaskTerminalKind.COMPLETED,
                    error={"code": "expected"} if fail else None,
                    result=None if fail else {"done": True},
                )
                terminal = await store.terminalize_task(terminal_request)
            elif fail:
                await store.fail_task(
                    identity,
                    {"code": "expected"},
                    worker_id="second",
                    lease_expires_at=second.lease_expires_at,
                )
            else:
                await store.complete_task(
                    identity,
                    {"done": True},
                    worker_id="second",
                    lease_expires_at=second.lease_expires_at,
                )
        finally:
            await store.close()
        reopened = SchedulingStorage(postgres_dsn, schema_mode=SchemaMode.VALIDATE)
        try:
            if receipt:
                assert await reopened.terminalize_task(terminal_request) == terminal
            assert [event.type for event in await reopened.list_task_schedule_events(identity)] == [
                TaskScheduleEventType.SCHEDULED,
                TaskScheduleEventType.HELD,
                TaskScheduleEventType.RESUMED,
                TaskScheduleEventType.ELIGIBLE,
                TaskScheduleEventType.CLAIMED,
                TaskScheduleEventType.RESUMED,
                TaskScheduleEventType.CLAIMED,
                TaskScheduleEventType.STARTED,
                TaskScheduleEventType.FAILED if fail else TaskScheduleEventType.COMPLETED,
            ]
        finally:
            await reopened.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    ("backend", "journal_failure"),
    [("memory", False), ("sqlite", False), ("postgres", False), ("sqlite", True)],
)
def test_scheduled_attached_failure_recovery_preserves_terminal_history(
    tmp_path, postgres_dsn, backend, journal_failure, monkeypatch
):
    async def run():
        identity = f"schedule-attached-recovery-{backend}"
        path = tmp_path / "attached-recovery.sqlite"
        if backend == "memory":
            store = InMemoryTaskStore()
        elif backend == "sqlite":
            store = SQLiteTaskStore(path)
        else:
            store = PostgresTaskStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            app = CayuApp(task_store=store)
            await app.create_task(
                TaskCreate(
                    task_id=identity,
                    type=identity,
                    available_at=datetime.now(UTC),
                    schedule_policy=TaskSchedulePolicy(),
                )
            )
            claimed = await store.claim_task(
                "lost-worker", TaskQuery(type=identity), lease_seconds=1
            )
            assert claimed is not None and claimed.lease_expires_at is not None
            attached = await store.attach_task(
                identity,
                session_id=f"session-{identity}",
                worker_id="lost-worker",
                lease_expires_at=claimed.lease_expires_at,
                session_invocation=await task_backed_session_invocation(
                    store, identity, f"session-{identity}"
                ),
            )
            assert attached.session_id is not None and attached.session_instance_id is not None
            request = TaskTerminalizationRequest(
                task_id=identity,
                worker_id="lost-worker",
                lease_expires_at=claimed.lease_expires_at,
                idempotency_key="recovery-failure",
                kind=TaskTerminalKind.FAILED,
                error={"code": "session_recovery"},
            )
            await asyncio.sleep(1.05)
            if journal_failure:
                assert isinstance(store, SQLiteTaskStore)
                record_transition = store._record_schedule_transition_unlocked

                def fail_after_journal_write(previous, current, *, operation_id=None):
                    record_transition(previous, current, operation_id=operation_id)
                    raise RuntimeError("injected journal publication failure")

                with monkeypatch.context() as patch:
                    patch.setattr(
                        store, "_record_schedule_transition_unlocked", fail_after_journal_write
                    )
                    with pytest.raises(RuntimeError, match="injected journal publication failure"):
                        await store.recover_attached_task_failure(
                            request,
                            session_id=attached.session_id,
                            session_instance_id=attached.session_instance_id,
                        )
                assert await store.load_task(identity) == attached
                assert (
                    await store.load_task_terminalization_receipt(identity, request.idempotency_key)
                    is None
                )
                assert all(
                    event.type is not TaskScheduleEventType.FAILED
                    for event in await app.list_task_schedule_events(identity)
                )
            terminal = await store.recover_attached_task_failure(
                request,
                session_id=attached.session_id,
                session_instance_id=attached.session_instance_id,
            )
            assert terminal.status is TaskStatus.FAILED
            if isinstance(store, SQLiteTaskStore):
                await store.close()
                store = SQLiteTaskStore(path)
            elif isinstance(store, PostgresTaskStore):
                await store.close()
                store = PostgresTaskStore(postgres_dsn, schema_mode=SchemaMode.VALIDATE)
            assert (
                await store.recover_attached_task_failure(
                    request,
                    session_id=attached.session_id,
                    session_instance_id=attached.session_instance_id,
                )
                == terminal
            )
            receipt = await store.load_task_terminalization_receipt(
                identity, request.idempotency_key
            )
            assert receipt is not None and receipt.task == terminal
            history = await CayuApp(task_store=store).list_task_schedule_events(identity)
            assert [event.type for event in history] == [
                TaskScheduleEventType.SCHEDULED,
                TaskScheduleEventType.ELIGIBLE,
                TaskScheduleEventType.CLAIMED,
                TaskScheduleEventType.STARTED,
                TaskScheduleEventType.FAILED,
            ]
        finally:
            if isinstance(store, (SQLiteTaskStore, PostgresTaskStore)):
                await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_rescheduling_first_retry_attempt_preserves_original_envelope(
    tmp_path, postgres_dsn, backend
):
    async def run():
        clock = [datetime.now(UTC)]
        initial_time = clock[0]
        identity = f"reschedule-retry-{backend}"
        if backend == "memory":
            store = InMemoryTaskStore(clock=lambda: clock[0], ownership_clock=lambda: clock[0])
        elif backend == "sqlite":
            store = SQLiteTaskStore(tmp_path / "retry-reschedule.sqlite", clock=lambda: clock[0])
        else:
            store = PostgresTaskStore(
                postgres_dsn, schema_mode=SchemaMode.CREATE, clock=lambda: clock[0]
            )
        assert store.supports_task_scheduling is True
        try:
            created = await store.create_task(
                TaskCreate(
                    task_id=identity,
                    type=identity,
                    available_at=initial_time + timedelta(seconds=1),
                    schedule_policy=TaskSchedulePolicy(),
                    retry_policy=TaskRetryPolicy(
                        max_attempts=3, max_elapsed_seconds=5, max_total_tokens=10
                    ),
                )
            )
            clock[0] += timedelta(seconds=2)
            request = TaskRescheduleRequest(
                task_id=identity,
                operation_id="delay",
                expected_revision=1,
                available_at=initial_time + timedelta(seconds=10),
            )
            receipt = await store.reschedule_task(request)
            updated = await store.load_task(identity)
            assert updated is not None and updated.retry_series is not None
            assert created.retry_series is not None
            assert updated.retry_series.authority_sha256 != created.retry_series.authority_sha256
            assert (
                updated.retry_series.model_copy(
                    update={"authority_sha256": created.retry_series.authority_sha256}
                )
                == created.retry_series
            )
            assert await store.reschedule_task(request) == receipt
            clock[0] = initial_time + timedelta(seconds=6)
            assert await store.claim_task("worker", TaskQuery(type=identity)) is None
            expired = await store.load_task(identity)
            assert expired is not None and expired.status is TaskStatus.FAILED
            assert expired.retry_series is not None
            assert expired.retry_series.elapsed_deadline == created.retry_series.elapsed_deadline
            assert expired.retry_series.cumulative_tokens == 0
            assert expired.retry_series.successor_task_id is None
        finally:
            if isinstance(store, (SQLiteTaskStore, PostgresTaskStore)):
                await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("persistent", [False, True])
def test_worker_wakes_at_persisted_due_time_before_long_poll(tmp_path, persistent):
    async def run():
        path = tmp_path / "worker.sqlite"
        store = SQLiteTaskStore(path) if persistent else InMemoryTaskStore()
        due = datetime.now(UTC) + timedelta(seconds=1)
        await store.create_task(
            TaskCreate(
                task_id="followup",
                type="followup",
                available_at=due,
                schedule_policy=TaskSchedulePolicy(),
            )
        )
        if isinstance(store, SQLiteTaskStore):
            await store.close()
            store = SQLiteTaskStore(path)
        app = CayuApp(task_store=store, enable_logging=False)
        calls = []

        async def handler(_app, task, worker_id):
            calls.append(datetime.now(UTC))
            assert calls[-1] >= due
            await complete_managed_task(store, task, worker_id, {"done": True})

        try:
            handled = await asyncio.wait_for(
                run_task_worker(
                    app,
                    store,
                    handler,
                    worker_id="worker",
                    query=TaskQuery(type="followup"),
                    poll_interval_s=10,
                    minimum_idle_delay_s=10,
                    maximum_idle_delay_s=10,
                    idle_jitter_ratio=0,
                    reclaim=False,
                    recover_interrupted_handoffs=False,
                    max_tasks=1,
                ),
                timeout=4,
            )
            assert handled == 1 and len(calls) == 1
            assert (await store.load_task("followup")).status is TaskStatus.COMPLETED
            events = await store.list_task_schedule_events("followup")
            assert sum(event.type is TaskScheduleEventType.CLAIMED for event in events) == 1
            assert events[-1].type is TaskScheduleEventType.COMPLETED
        finally:
            if isinstance(store, SQLiteTaskStore):
                await store.close()

    asyncio.run(run())


def test_postgres_task_projections_have_identical_column_order():
    from cayu.storage import _postgres_support
    from cayu.storage.postgres import _TASK_RETURNING_COLUMNS

    assert _TASK_RETURNING_COLUMNS.split(", ") == [
        f"task.{column}" for column in _postgres_support.TASK_COLUMNS.split(", ")
    ]


def test_task_schedule_policy_copies_mutated_models_without_serializer_warnings(recwarn):
    class PrivateValue:
        def __repr__(self):
            return "private-schedule-canary"

    policy = TaskSchedulePolicy()
    object.__setattr__(policy, "misfire_grace_seconds", PrivateValue())
    with pytest.raises(ValidationError) as raised:
        copy_task_schedule_policy(policy)
    assert "private-schedule-canary" not in str(raised.value)
    assert not recwarn


def test_task_schedule_decision_does_not_overflow_latest_datetime():
    due = datetime.max.replace(tzinfo=UTC)
    assert (
        task_schedule_eligibility(available_at=due, policy=TaskSchedulePolicy(), as_of=due)
        is TaskScheduleEligibility.ELIGIBLE
    )


def test_memory_public_creation_reschedule_replay_and_claim():
    async def run():
        clock = [datetime(2026, 9, 14, tzinfo=UTC)]
        store = InMemoryTaskStore(clock=lambda: clock[0])
        app = CayuApp(task_store=store)
        request = TaskCreate(
            task_id="followup",
            type="invoice",
            input={"invoice_id": "inv-1"},
            available_at=clock[0] + timedelta(hours=1),
            schedule_policy=TaskSchedulePolicy(),
        )
        created = await app.create_task(request)
        assert created.schedule.revision == 1
        assert await store.claim_task("worker") is None
        changed_request = TaskRescheduleRequest(
            task_id=created.id,
            operation_id="move",
            expected_revision=1,
            available_at=clock[0] + timedelta(hours=2),
        )
        receipt = await store.reschedule_task(changed_request)
        assert receipt.schedule.revision == 2
        replayed_create = await app.create_task(request)
        assert replayed_create.available_at == changed_request.available_at
        assert replayed_create.schedule.revision == 2
        with pytest.raises(TaskScheduleConflict):
            await app.create_task(request.model_copy(update={"input": {"invoice_id": "other"}}))
        clock[0] = changed_request.available_at
        claims = await asyncio.gather(store.claim_task("one"), store.claim_task("two"))
        claimed = next(task for task in claims if task is not None)
        assert sum(task is not None for task in claims) == 1
        assert claimed.schedule.revision == 3
        assert claimed.schedule.admitted_at == clock[0]
        assert await store.reschedule_task(changed_request) == receipt
        with pytest.raises(TaskScheduleConflict):
            await store.reschedule_task(
                changed_request.model_copy(update={"operation_id": "stale"})
            )
        events = await store.list_task_schedule_events(created.id)
        kinds = [event.type for event in events]
        assert [event.operation_id for event in events] == [None, "move", None, None]
        assert kinds == [
            TaskScheduleEventType.SCHEDULED,
            TaskScheduleEventType.RESCHEDULED,
            TaskScheduleEventType.ELIGIBLE,
            TaskScheduleEventType.CLAIMED,
        ]

    asyncio.run(run())


@pytest.mark.parametrize("persistent", [False, True])
def test_schedule_cancel_reconciliation_retains_journal_after_reconstruction(tmp_path, persistent):
    SchedulingStorage = SQLiteTaskStore

    async def run():
        clock = [datetime.now(UTC)]
        path = tmp_path / "cancel-reconciliation.sqlite"
        store = (
            SchedulingStorage(path, clock=lambda: clock[0], ownership_clock=lambda: clock[0])
            if persistent
            else InMemoryTaskStore(clock=lambda: clock[0], ownership_clock=lambda: clock[0])
        )
        try:
            await store.create_task(
                TaskCreate(
                    task_id="cancel-followup",
                    type="followup",
                    available_at=clock[0],
                    schedule_policy=TaskSchedulePolicy(),
                    metadata={
                        "execution_profile_fingerprint": "b" * 64,
                        "effect_fingerprint": "c" * 64,
                    },
                )
            )
            claimed = await store.claim_task("original", lease_seconds=1)
            assert claimed is not None and claimed.schedule is not None
            cancellation = TaskScheduleCancelRequest(
                task_id=claimed.id,
                operation_id="cancel",
                expected_revision=claimed.schedule.revision,
            )
            accepted = await store.cancel_scheduled_task(cancellation)
            requested = await store.load_task(claimed.id)
            assert requested is not None
            assert requested.status is TaskStatus.CLAIMED
            request = ordinary_cancellation_reconciliation_request(requested)
            clock[0] += timedelta(seconds=2)
            assert await store.claim_task("replacement") is None
            if isinstance(store, SQLiteTaskStore):
                await store.close()
                store = SchedulingStorage(
                    path, clock=lambda: clock[0], ownership_clock=lambda: clock[0]
                )
            result = await store.reconcile_task_cancellation(request)
            assert result.task.status is TaskStatus.CANCELLED
            assert result.task.schedule == requested.schedule
            assert await store.reconcile_task_cancellation(request) == result
            assert await store.cancel_scheduled_task(cancellation) == accepted
            assert [event.type for event in await store.list_task_schedule_events(claimed.id)] == [
                TaskScheduleEventType.SCHEDULED,
                TaskScheduleEventType.ELIGIBLE,
                TaskScheduleEventType.CLAIMED,
                TaskScheduleEventType.CANCELLATION_REQUESTED,
                TaskScheduleEventType.CANCELLED,
            ]
        finally:
            if isinstance(store, SQLiteTaskStore):
                await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("publication_failure", [False, True])
def test_postgres_schedule_create_and_reschedule_replay_after_reopen(
    postgres_dsn, publication_failure
):
    fail_publication = [publication_failure]

    class SchedulingStorage(PostgresTaskStore):
        async def _record_schedule_transition(self, cur, prior, current, *, operation_id=None):
            await super()._record_schedule_transition(
                cur, prior, current, operation_id=operation_id
            )
            if prior is not None and fail_publication[0]:
                fail_publication[0] = False
                raise RuntimeError("schedule journal acknowledgement lost before commit")

    async def run():
        due = datetime.now(UTC) + timedelta(hours=1)
        task_id = f"scheduled-reopen-{publication_failure}"
        creation = TaskCreate(
            task_id=task_id,
            type="followup",
            available_at=due,
            schedule_policy=TaskSchedulePolicy(),
        )
        request = TaskRescheduleRequest(
            task_id=task_id,
            operation_id="move",
            expected_revision=1,
            available_at=due + timedelta(hours=1),
        )
        store = SchedulingStorage(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            created = await store.create_task(creation)
            assert await store.create_task(creation) == created
            if publication_failure:
                with pytest.raises(RuntimeError, match="journal acknowledgement"):
                    await store.reschedule_task(request)
                assert await store.load_task(created.id) == created
                assert len(await store.list_task_schedule_events(created.id)) == 1
            receipt = await store.reschedule_task(request)
            current = await store.load_task(created.id)
            assert current is not None and current.schedule is not None
            assert current.available_at == request.available_at
            assert current.schedule.revision == 2
        finally:
            await store.close()
        reopened = SchedulingStorage(postgres_dsn, schema_mode=SchemaMode.VALIDATE)
        try:
            assert await reopened.create_task(creation) == current
            assert await reopened.reschedule_task(request) == receipt
            with pytest.raises(TaskScheduleConflict):
                await reopened.reschedule_task(request.model_copy(update={"available_at": due}))
            assert await reopened.load_task(created.id) == current
            assert [
                event.type for event in await reopened.list_task_schedule_events(created.id)
            ] == [
                TaskScheduleEventType.SCHEDULED,
                TaskScheduleEventType.RESCHEDULED,
            ]
        finally:
            await reopened.close()

    asyncio.run(run())


@pytest.mark.parametrize("misfire", list(TaskMisfirePolicy))
def test_memory_schedule_expiry_and_misfire_prevent_handler_admission(misfire):
    async def run():
        clock = [datetime(2026, 9, 14, tzinfo=UTC)]
        store = InMemoryTaskStore(clock=lambda: clock[0])
        due = clock[0] + timedelta(seconds=10)
        await store.create_task(
            TaskCreate(
                task_id="late",
                type="followup",
                available_at=due,
                schedule_policy=TaskSchedulePolicy(misfire_policy=misfire, misfire_grace_seconds=5),
            )
        )
        await store.create_task(
            TaskCreate(
                task_id="expired",
                type="followup",
                available_at=due,
                schedule_policy=TaskSchedulePolicy(expires_at=due + timedelta(seconds=5)),
            )
        )
        clock[0] = due + timedelta(seconds=6)
        wakeup = await store.next_task_schedule_wakeup()
        assert wakeup.maintenance_required
        claimed = await store.claim_task("worker")
        assert (claimed is not None) == (misfire is TaskMisfirePolicy.FIRE_ONCE)
        expired = await store.load_task("expired")
        assert expired.status is TaskStatus.CANCELLED
        assert expired.status_reason == "schedule_expired"
        assert (await store.list_task_schedule_events("expired"))[
            -1
        ].type is TaskScheduleEventType.EXPIRED
        if misfire is TaskMisfirePolicy.SKIP:
            assert (await store.load_task("late")).status_reason == "schedule_skipped"

    asyncio.run(run())


def test_memory_cancel_schedule_is_revision_fenced_and_retains_claim():
    async def run():
        now = datetime.now(UTC)
        store = InMemoryTaskStore(clock=lambda: now)
        task = await store.create_task(
            TaskCreate(
                task_id="cancel",
                type="followup",
                available_at=now,
                schedule_policy=TaskSchedulePolicy(),
            )
        )
        claimed = await store.claim_task("worker")
        with pytest.raises(TaskScheduleConflict):
            await store.cancel_scheduled_task(
                TaskScheduleCancelRequest(
                    task_id=task.id,
                    operation_id="stale",
                    expected_revision=1,
                )
            )
        with pytest.raises(TaskScheduleConflict):
            await store.cancel_task(task.id)
        request = TaskScheduleCancelRequest(
            task_id=task.id, operation_id="cancel", expected_revision=2
        )
        receipt = await store.cancel_scheduled_task(request)
        assert receipt.type is TaskScheduleEventType.CANCELLATION_REQUESTED
        current = await store.load_task(task.id)
        assert current.worker_id == claimed.worker_id
        assert current.lease_expires_at == claimed.lease_expires_at
        assert current.status is TaskStatus.CLAIMED
        assert await store.cancel_scheduled_task(request) == receipt
        events = await store.list_task_schedule_events(task.id)
        assert events[-1].operation_id == request.operation_id
        assert events[-1].type is TaskScheduleEventType.CANCELLATION_REQUESTED
        assert await store.claim_task("replacement") is None

    asyncio.run(run())


@pytest.mark.parametrize("claimed", [False, True])
@pytest.mark.parametrize("retry", [False, True])
def test_memory_schedule_cancel_preparation_failure_publishes_nothing(monkeypatch, claimed, retry):
    from cayu.tasks import base

    async def run():
        now = datetime.now(UTC)
        store = InMemoryTaskStore(clock=lambda: now)
        await store.create_task(
            TaskCreate(
                task_id="cancel-atomic",
                type="followup",
                available_at=now,
                schedule_policy=TaskSchedulePolicy(),
                retry_policy=TaskRetryPolicy(max_attempts=2) if retry else None,
            )
        )
        if claimed:
            await store.claim_task("worker")
        before = await store.load_task("cancel-atomic")
        events_before = await store.list_task_schedule_events(before.id)
        request = TaskScheduleCancelRequest(
            task_id=before.id, operation_id="cancel", expected_revision=before.schedule.revision
        )

        def fail_receipt(*args, **kwargs):
            raise ValueError("receipt preparation failed")

        with monkeypatch.context() as patch:
            patch.setattr(base, "schedule_receipt", fail_receipt)
            with pytest.raises(ValueError, match="receipt preparation failed"):
                await store.cancel_scheduled_task(request)
        assert await store.load_task(before.id) == before
        assert await store.list_task_schedule_events(before.id) == events_before
        assert not store._schedule_receipts
        assert not store._retry_settlements
        receipt = await store.cancel_scheduled_task(request)
        assert receipt.schedule.revision == before.schedule.revision + 1
        events = await store.list_task_schedule_events(before.id)
        assert len(events) == len(events_before) + 1
        assert events[-1].operation_id == request.operation_id
        assert await store.cancel_scheduled_task(request) == receipt
        if retry and not claimed:
            settlement = next(iter(store._retry_settlements.values()))
            assert settlement.task == await store.load_task(before.id)

    asyncio.run(run())


@pytest.mark.parametrize("expiry", [False, True])
def test_memory_schedule_nonexecution_settles_retry_authority(expiry):
    async def run():
        clock = [datetime.now(UTC)]
        store = InMemoryTaskStore(clock=lambda: clock[0])
        due = clock[0] + timedelta(seconds=10)
        await store.create_task(
            TaskCreate(
                task_id="never-dispatched",
                type="followup",
                available_at=due,
                schedule_policy=TaskSchedulePolicy(
                    expires_at=due + timedelta(seconds=1) if expiry else None,
                    misfire_policy=TaskMisfirePolicy.FIRE_ONCE
                    if expiry
                    else TaskMisfirePolicy.SKIP,
                    misfire_grace_seconds=0,
                ),
                retry_policy=TaskRetryPolicy(max_attempts=2),
            )
        )
        clock[0] = due + timedelta(seconds=2)
        assert await store.claim_task("worker") is None
        task = await store.load_task("never-dispatched")
        assert task.status is TaskStatus.CANCELLED
        assert task.retry_series.disposition == "cancelled"
        assert task.retry_series.successor_task_id is None
        assert task.retry_series.cumulative_tokens == 0
        receipt = next(iter(store._retry_settlements.values()))
        assert receipt.task == task
        assert receipt.successor is None
        assert (await store.list_task_schedule_events(task.id))[-1].type is (
            TaskScheduleEventType.EXPIRED if expiry else TaskScheduleEventType.SKIPPED
        )
        assert await store.claim_task("retry-worker") is None
        assert len(store._retry_settlements) == 1

    asyncio.run(run())


@pytest.mark.parametrize("retry", [False, True])
def test_sqlite_schedule_mutations_reconstruct_exact_receipts(tmp_path, retry):
    # Storage-transaction characterization while the complete SQLite scheduling
    # capability remains gated pending claim/worker integration.
    SchedulingStorage = SQLiteTaskStore

    async def run():
        now = datetime.now(UTC)
        path = tmp_path / "scheduled.sqlite"
        request = TaskCreate(
            task_id="persisted-followup",
            type="followup",
            available_at=now + timedelta(hours=1),
            schedule_policy=TaskSchedulePolicy(),
            retry_policy=TaskRetryPolicy(max_attempts=2) if retry else None,
        )
        store = SchedulingStorage(path, clock=lambda: now)
        try:
            task = await store.create_task(request)
            assert task.schedule.revision == 1
            if not retry:
                move = TaskRescheduleRequest(
                    task_id=task.id,
                    operation_id="move",
                    expected_revision=1,
                    available_at=now + timedelta(hours=2),
                )
                moved = await store.reschedule_task(move)
                assert moved.schedule.revision == 2
            current = await store.load_task(task.id)
            cancel = TaskScheduleCancelRequest(
                task_id=task.id, operation_id="cancel", expected_revision=current.schedule.revision
            )
            cancelled = await store.cancel_scheduled_task(cancel)
            assert cancelled.type is TaskScheduleEventType.CANCELLED
        finally:
            await store.close()
        reopened = SchedulingStorage(path, clock=lambda: now)
        try:
            assert await reopened.cancel_scheduled_task(cancel) == cancelled
            if not retry:
                assert await reopened.reschedule_task(move) == moved
            current = await reopened.create_task(request)
            assert current.status is TaskStatus.CANCELLED
            assert current.schedule == cancelled.schedule
            events = await reopened.list_task_schedule_events(task.id)
            assert [event.operation_id for event in events] == (
                [None, "cancel"] if retry else [None, "move", "cancel"]
            )
            with pytest.raises(TaskScheduleConflict):
                await reopened.cancel_scheduled_task(
                    cancel.model_copy(update={"expected_revision": 99})
                )
        finally:
            await reopened.close()

    asyncio.run(run())


@pytest.mark.parametrize("cancel", [False, True])
def test_sqlite_schedule_journal_failure_rolls_back_task_and_receipt(tmp_path, cancel):
    class SchedulingStorage(SQLiteTaskStore):
        fail_journal = False

        def _record_schedule_transition_unlocked(self, prior, current, *, operation_id=None):
            super()._record_schedule_transition_unlocked(prior, current, operation_id=operation_id)
            if self.fail_journal:
                raise OSError("injected journal publication failure")

    async def run():
        now = datetime.now(UTC)
        path = tmp_path / "atomic.sqlite"
        store = SchedulingStorage(path, clock=lambda: now)
        try:
            initial = await store.create_task(
                TaskCreate(
                    task_id="atomic",
                    type="followup",
                    available_at=now + timedelta(hours=1),
                    schedule_policy=TaskSchedulePolicy(),
                )
            )
            store.fail_journal = True
            with pytest.raises(OSError, match="journal publication failure"):
                if cancel:
                    await store.cancel_scheduled_task(
                        TaskScheduleCancelRequest(
                            task_id=initial.id, operation_id="edit", expected_revision=1
                        )
                    )
                else:
                    await store.reschedule_task(
                        TaskRescheduleRequest(
                            task_id=initial.id,
                            operation_id="edit",
                            expected_revision=1,
                            available_at=now + timedelta(hours=2),
                        )
                    )
        finally:
            await store.close()
        reopened = SchedulingStorage(path, clock=lambda: now)
        try:
            assert await reopened.load_task(initial.id) == initial
            assert len(await reopened.list_task_schedule_events(initial.id)) == 1
            # Reuse the operation identity for different content: a failed
            # transaction must not have retained an idempotency binding.
            changed = await reopened.reschedule_task(
                TaskRescheduleRequest(
                    task_id=initial.id,
                    operation_id="edit",
                    expected_revision=1,
                    available_at=now + timedelta(hours=3),
                )
            )
            assert changed.schedule.revision == 2
        finally:
            await reopened.close()

    asyncio.run(run())


@pytest.mark.parametrize("late", [False, True])
def test_sqlite_schedule_claim_reopening_and_competing_workers(tmp_path, late):
    SchedulingStorage = SQLiteTaskStore

    async def run():
        clock = [datetime.now(UTC)]
        due = clock[0] + timedelta(hours=1)
        path = tmp_path / "claim.sqlite"
        creator = SchedulingStorage(path, clock=lambda: clock[0])
        try:
            for identity, policy in (
                ("expired", TaskSchedulePolicy(expires_at=due + timedelta(seconds=1))),
                (
                    "skip",
                    TaskSchedulePolicy(
                        misfire_policy=TaskMisfirePolicy.SKIP, misfire_grace_seconds=0
                    ),
                ),
                ("once", TaskSchedulePolicy(misfire_grace_seconds=0)),
            ):
                await creator.create_task(
                    TaskCreate(
                        task_id=identity,
                        type="followup",
                        available_at=due,
                        schedule_policy=policy,
                    )
                )
            assert await creator.claim_task("too-early") is None
        finally:
            await creator.close()
        clock[0] = due + timedelta(seconds=2) if late else due
        first = SchedulingStorage(path, clock=lambda: clock[0])
        second = SchedulingStorage(path, clock=lambda: clock[0])
        try:
            results = await asyncio.gather(first.claim_task("one"), second.claim_task("two"))
            claimed = [task for task in results if task is not None]
            assert len({task.id for task in claimed}) == len(claimed)
            assert len(claimed) == (1 if late else 2)
            for task in claimed:
                assert task.schedule.revision == 2
                assert task.schedule.admitted_at == clock[0]
                with pytest.raises(TaskScheduleConflict):
                    await first.reschedule_task(
                        TaskRescheduleRequest(
                            task_id=task.id,
                            operation_id="late-edit",
                            expected_revision=1,
                            available_at=due + timedelta(hours=2),
                        )
                    )
                with pytest.raises(TaskScheduleConflict):
                    await first.cancel_task(task.id)
                events = await first.list_task_schedule_events(task.id)
                assert events[-2].type is (
                    TaskScheduleEventType.MISFIRED if late else TaskScheduleEventType.ELIGIBLE
                )
                assert events[-1].type is TaskScheduleEventType.CLAIMED
            if late:
                assert claimed[0].id == "once"
                assert (await first.load_task("expired")).status_reason == "schedule_expired"
                assert (await first.load_task("skip")).status_reason == "schedule_skipped"
        finally:
            await first.close()
            await second.close()

    asyncio.run(run())


def test_sqlite_schedule_expiry_batches_do_not_strand_later_work(tmp_path):
    SchedulingStorage = SQLiteTaskStore

    async def run():
        clock = [datetime.now(UTC)]
        due = clock[0] + timedelta(seconds=1)
        store = SchedulingStorage(tmp_path / "batches.sqlite", clock=lambda: clock[0])
        try:
            for index in range(101):
                await store.create_task(
                    TaskCreate(
                        task_id=f"expired-{index:03}",
                        type="followup",
                        available_at=due,
                        schedule_policy=TaskSchedulePolicy(expires_at=due + timedelta(seconds=1)),
                    )
                )
            await store.create_task(
                TaskCreate(
                    task_id="surviving",
                    type="followup",
                    available_at=due,
                    schedule_policy=TaskSchedulePolicy(),
                )
            )
            clock[0] = due + timedelta(seconds=2)
            assert await store.claim_task("batch-one") is None
            winner = await store.claim_task("batch-two")
            assert winner.id == "surviving"
            assert await store.claim_task("batch-three") is None
            for index in range(101):
                events = await store.list_task_schedule_events(f"expired-{index:03}")
                assert [event.type for event in events] == [
                    TaskScheduleEventType.SCHEDULED,
                    TaskScheduleEventType.EXPIRED,
                ]
        finally:
            await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_schedule_wakeup_matches_claim_filters_and_ignores_past_and_held_due(tmp_path, backend):
    SchedulingStorage = SQLiteTaskStore

    async def run():
        now = datetime.now(UTC)
        store = (
            InMemoryTaskStore(clock=lambda: now)
            if backend == "memory"
            else SchedulingStorage(tmp_path / "wake.sqlite", clock=lambda: now)
        )
        try:
            for identity, kind, agent, minutes in (
                ("past", "followup", "agent", -1),
                ("other-type", "other", "agent", 1),
                ("other-agent", "followup", "other", 2),
                ("held", "followup", "agent", 3),
                ("future", "followup", "agent", 30),
            ):
                await store.create_task(
                    TaskCreate(
                        task_id=identity,
                        type=kind,
                        assigned_agent_name=agent,
                        available_at=now + timedelta(minutes=minutes),
                        schedule_policy=TaskSchedulePolicy(),
                    )
                )
            await store.pause_task("held")
            query = TaskQuery(type="followup", assigned_agent_name="agent")
            wake = await store.next_task_schedule_wakeup(query)
            assert wake.as_of == now
            assert wake.next_available_at == now + timedelta(minutes=30)
            assert not wake.maintenance_required
            excluded = await store.next_task_schedule_wakeup(
                query.model_copy(update={"status": TaskStatus.COMPLETED})
            )
            assert excluded.next_available_at is None
            await store.reschedule_task(
                TaskRescheduleRequest(
                    task_id="future",
                    operation_id="earlier",
                    expected_revision=1,
                    available_at=now + timedelta(minutes=10),
                )
            )
            assert (
                await store.next_task_schedule_wakeup(query)
            ).next_available_at == now + timedelta(minutes=10)
            await store.cancel_scheduled_task(
                TaskScheduleCancelRequest(
                    task_id="future",
                    operation_id="cancel",
                    expected_revision=2,
                )
            )
            assert (await store.next_task_schedule_wakeup(query)).next_available_at is None
        finally:
            if backend == "sqlite":
                await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("misfire", list(TaskMisfirePolicy))
def test_held_schedule_maintenance_settles_without_dispatch(tmp_path, backend, misfire):
    SchedulingStorage = SQLiteTaskStore

    async def run():
        clock = [datetime.now(UTC)]
        due = clock[0] + timedelta(seconds=1)
        store = (
            InMemoryTaskStore(clock=lambda: clock[0])
            if backend == "memory"
            else SchedulingStorage(tmp_path / "held.sqlite", clock=lambda: clock[0])
        )
        try:
            await store.create_task(
                TaskCreate(
                    task_id="held",
                    type="followup",
                    available_at=due,
                    schedule_policy=TaskSchedulePolicy(
                        misfire_policy=misfire,
                        misfire_grace_seconds=0,
                        expires_at=due + timedelta(seconds=1)
                        if misfire is TaskMisfirePolicy.FIRE_ONCE
                        else None,
                    ),
                    retry_policy=TaskRetryPolicy(max_attempts=2),
                )
            )
            await store.pause_task("held")
            clock[0] = due + timedelta(seconds=2)
            assert (
                await store.next_task_schedule_wakeup(TaskQuery(type="followup"))
            ).maintenance_required
            assert await store.claim_task("unrelated", TaskQuery(type="other")) is None
            assert (await store.load_task("held")).status is TaskStatus.PAUSED
            assert await store.claim_task("worker", TaskQuery(type="followup")) is None
            task = await store.load_task("held")
            assert task.status is TaskStatus.CANCELLED
            assert task.retry_series.disposition == "cancelled"
            wake = await store.next_task_schedule_wakeup(TaskQuery(type="followup"))
            assert not wake.maintenance_required
            assert wake.next_available_at is None
            assert wake.next_expiry_at is None
        finally:
            if backend == "sqlite":
                await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("receipt", [False, True])
@pytest.mark.parametrize("hold_method", ["pause_task", "block_task", "mark_task_needs_attention"])
def test_sqlite_schedule_terminal_evidence_survives_reopening(
    tmp_path, failed, receipt, hold_method
):
    SchedulingStorage = SQLiteTaskStore

    async def run():
        now = datetime.now(UTC)
        path = tmp_path / "terminal.sqlite"
        store = SchedulingStorage(path, clock=lambda: now)
        try:
            await store.create_task(
                TaskCreate(
                    task_id="terminal",
                    type="followup",
                    available_at=now,
                    schedule_policy=TaskSchedulePolicy(),
                )
            )
            await getattr(store, hold_method)("terminal")
            await store.resume_task("terminal")
            task = await store.claim_task("worker")
            started = await store.mark_claimed_task_execution_started(
                task.id, "worker", task.lease_expires_at
            )
            assert (
                await store.mark_claimed_task_execution_started(
                    task.id, "worker", task.lease_expires_at
                )
                == started
            )
            if receipt:
                terminal_request = TaskTerminalizationRequest(
                    task_id=task.id,
                    worker_id="worker",
                    lease_expires_at=task.lease_expires_at,
                    kind=TaskTerminalKind.FAILED if failed else TaskTerminalKind.COMPLETED,
                    result=None if failed else {"done": True},
                    error={"code": "downstream_failed"} if failed else None,
                    idempotency_key="terminal",
                )
                terminal = await store.terminalize_task(terminal_request)
            elif failed:
                await store.fail_task(
                    task.id,
                    {"code": "downstream_failed"},
                    worker_id="worker",
                    lease_expires_at=task.lease_expires_at,
                )
            else:
                await store.complete_task(
                    task.id,
                    {"done": True},
                    worker_id="worker",
                    lease_expires_at=task.lease_expires_at,
                )
        finally:
            await store.close()
        reopened = SchedulingStorage(path, clock=lambda: now)
        try:
            if receipt:
                assert await reopened.terminalize_task(terminal_request) == terminal
            events = await reopened.list_task_schedule_events("terminal")
            assert [event.type for event in events] == [
                TaskScheduleEventType.SCHEDULED,
                TaskScheduleEventType.HELD,
                TaskScheduleEventType.RESUMED,
                TaskScheduleEventType.ELIGIBLE,
                TaskScheduleEventType.CLAIMED,
                TaskScheduleEventType.STARTED,
                TaskScheduleEventType.FAILED if failed else TaskScheduleEventType.COMPLETED,
            ]
            assert await reopened.claim_task("replacement") is None
        finally:
            await reopened.close()

    asyncio.run(run())


@pytest.mark.parametrize("reclaim", [False, True])
def test_sqlite_schedule_worker_handoff_keeps_first_admission(tmp_path, reclaim):
    SchedulingStorage = SQLiteTaskStore

    async def run():
        clock = [datetime.now(UTC)]
        store = SchedulingStorage(
            tmp_path / "handoff.sqlite", clock=lambda: clock[0], ownership_clock=lambda: clock[0]
        )
        try:
            await store.create_task(
                TaskCreate(
                    task_id="handoff",
                    type="followup",
                    available_at=clock[0],
                    schedule_policy=TaskSchedulePolicy(expires_at=clock[0] + timedelta(seconds=1)),
                )
            )
            first = await store.claim_task("first", lease_seconds=1 if reclaim else 60)
            clock[0] += timedelta(seconds=2)
            if reclaim:
                reclaimed = await store.reclaim_expired()
                assert [task.id for task in reclaimed] == [first.id]
            else:
                await store.release_task(first.id, "first", lease_expires_at=first.lease_expires_at)
            second = await store.claim_task("second")
            assert second.schedule == first.schedule
            assert second.worker_id == "second"
            attached = await store.attach_task(
                second.id,
                session_id="followup-session",
                worker_id="second",
                lease_expires_at=second.lease_expires_at,
                session_invocation=await task_backed_session_invocation(
                    store, second.id, "followup-session"
                ),
            )
            assert attached.schedule == first.schedule
            assert attached.status is TaskStatus.RUNNING
            assert [event.type for event in await store.list_task_schedule_events(first.id)] == [
                TaskScheduleEventType.SCHEDULED,
                TaskScheduleEventType.ELIGIBLE,
                TaskScheduleEventType.CLAIMED,
                TaskScheduleEventType.RESUMED,
                TaskScheduleEventType.CLAIMED,
                TaskScheduleEventType.STARTED,
            ]
        finally:
            await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("deadline", ["none", "pending", "claimed"])
def test_sqlite_schedule_retry_terminal_evidence(tmp_path, deadline):
    SchedulingStorage = SQLiteTaskStore

    async def run():
        clock = [datetime.now(UTC)]
        path = tmp_path / "retry.sqlite"
        store = SchedulingStorage(path, clock=lambda: clock[0], ownership_clock=lambda: clock[0])
        try:
            await store.create_task(
                TaskCreate(
                    task_id="retry",
                    type="followup",
                    available_at=clock[0],
                    schedule_policy=TaskSchedulePolicy(),
                    retry_policy=TaskRetryPolicy(
                        max_attempts=2, max_elapsed_seconds=1.0 if deadline != "none" else None
                    ),
                )
            )
            if deadline == "claimed":
                claimed = await store.claim_task("worker")
                clock[0] += timedelta(seconds=2)
                assert (
                    await store.enforce_task_retry_deadline(
                        claimed.id, "worker", lease_expires_at=claimed.lease_expires_at
                    )
                    is not None
                )
            elif deadline == "pending":
                clock[0] += timedelta(seconds=2)
                assert await store.claim_task("worker") is None
            else:
                claimed = await store.claim_task("worker")
                request = TaskRetrySettlementRequest(
                    task_id=claimed.id,
                    worker_id="worker",
                    lease_expires_at=claimed.lease_expires_at,
                    idempotency_key="settle",
                    causal_budget_id=claimed.retry_series.causal_budget_id,
                    disposition=TaskRetryAttemptDisposition.SUCCEEDED,
                    result={"done": True},
                )
                receipt = await store.settle_task_retry_attempt(request)
        finally:
            await store.close()
        reopened = SchedulingStorage(path, clock=lambda: clock[0])
        try:
            if deadline == "none":
                assert await reopened.settle_task_retry_attempt(request) == receipt
            events = await reopened.list_task_schedule_events("retry")
            assert events[-1].type is (
                TaskScheduleEventType.FAILED
                if deadline != "none"
                else TaskScheduleEventType.COMPLETED
            )
            assert len(events) == (2 if deadline == "pending" else 4)
        finally:
            await reopened.close()

    asyncio.run(run())
