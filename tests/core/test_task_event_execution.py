"""Event identity and execution deadlines survive task replacement and restart."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from cayu import CayuApp
from cayu.sessions.invocation import InvocationOrigin
from cayu.storage import SQLiteTaskStore
from cayu.tasks import (
    InMemoryTaskStore,
    TaskCreate,
    TaskRetryBudget,
    TaskStatus,
    run_task_worker,
)
from cayu.tasks.retry_budgets import (
    TaskRetryBudgetExhausted,
    TaskRetryKeyStatus,
    reconcile_task_event_deadlines,
)


def _store(backend, tmp_path, request, *, clock=None):
    kwargs = {} if clock is None else {"clock": clock}
    if backend == "memory":
        return InMemoryTaskStore(**kwargs)
    if backend == "sqlite":
        return SQLiteTaskStore(tmp_path / "events.sqlite", **kwargs)
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.tasks_postgres import PostgresTaskStore

    return PostgresTaskStore(
        request.getfixturevalue("event_postgres_dsn"), schema_mode=SchemaMode.CREATE, **kwargs
    )


@pytest.fixture
def event_postgres_dsn(_postgres_server_dsn):
    from uuid import uuid4

    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    database = f"cayu_event_{uuid4().hex}"
    with psycopg.connect(_postgres_server_dsn, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
    try:
        yield make_conninfo(_postgres_server_dsn, dbname=database)
    finally:
        with psycopg.connect(_postgres_server_dsn, autocommit=True) as connection:
            connection.execute(
                sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(database))
            )


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_event_identity_uses_event_id_not_payload_or_attempt_metadata(backend, tmp_path, request):
    async def scenario():
        store = _store(backend, tmp_path, request)
        first = await store.create_task(
            TaskCreate(type="job", input={"event_id": "one", "attempt": 1})
        )
        second = await store.create_task(
            TaskCreate(type="job", input={"event_id": "one", "attempt": 2})
        )
        independent = await store.create_task(
            TaskCreate(type="job", event_id="two", input={"event_id": "one", "attempt": 1})
        )
        plain_one = await store.create_task(TaskCreate(type="job", input={"value": 1}))
        plain_two = await store.create_task(TaskCreate(type="job", input={"value": 1}))
        assert first.invocation.event_retry_key == second.invocation.event_retry_key
        assert independent.invocation.event_retry_key != first.invocation.event_retry_key
        assert plain_one.invocation.event_retry_key != plain_two.invocation.event_retry_key
        state = await store.load_task_retry_key(first.invocation.event_retry_key)
        assert state.attempts == 2

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_last_failure_persists_exhaustion_without_another_submission(backend, tmp_path, request):
    async def scenario():
        store = _store(backend, tmp_path, request)
        for _index in range(3):
            task = await store.create_task(TaskCreate(type="job", event_id="one"))
            await store.fail_task(task.id, {"code": "failed"})
        state = await store.load_task_retry_key(task.invocation.event_retry_key)
        assert state.status is TaskRetryKeyStatus.GIVEN_UP
        assert state.given_up_reason.value == "attempts_exhausted"
        assert state.settled_attempts == 3
        assert state.refused_task_id is None

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_admission_exhaustion_does_not_cancel_running_attempts(backend, tmp_path, request):
    async def scenario():
        store = _store(backend, tmp_path, request)
        tasks = await asyncio.gather(
            *(store.create_task(TaskCreate(type="job", event_id="one")) for _ in range(3))
        )
        with pytest.raises(TaskRetryBudgetExhausted):
            await store.create_task(TaskCreate(type="job", event_id="one"))
        state = await store.refresh_task_retry_key(tasks[0].invocation.event_retry_key)
        assert state.status is TaskRetryKeyStatus.ACTIVE
        assert await store.expired_task_event_ids() == []

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_hung_active_handler_is_cancelled_by_event_deadline(backend, tmp_path, request):
    async def scenario():
        store = _store(backend, tmp_path, request)
        app = CayuApp(task_store=store, enable_logging=False)
        task = await app.create_task(
            TaskCreate(
                type="job",
                event_id="one",
                # Leave admission time; this test exercises cancellation of an active handler.
                retry_budget=TaskRetryBudget(max_attempts=None, max_elapsed_seconds=1.0),
            )
        )
        started, stopped = asyncio.Event(), asyncio.Event()

        async def handler(app, task, worker_id):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        try:
            await asyncio.wait_for(
                run_task_worker(app, store, handler, worker_id="worker", max_tasks=1), 5
            )
            assert started.is_set() and stopped.is_set()
            final = await store.load_task(task.id)
            assert final.status is TaskStatus.FAILED
            state = await store.load_task_retry_key(task.invocation.event_retry_key)
            assert state.status is TaskRetryKeyStatus.GIVEN_UP
            assert state.given_up_reason.value == "elapsed_exhausted"
            assert final.error["code"] == "event_elapsed_exhausted"
        finally:
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_idle_event_expires_and_unlimited_work_remains_available(backend, tmp_path, request):
    now = [datetime(2026, 10, 27, tzinfo=UTC)]

    async def scenario():
        store = _store(backend, tmp_path, request, clock=lambda: now[0])
        bounded = await store.create_task(
            TaskCreate(
                type="job", event_id="bounded", retry_budget=TaskRetryBudget(max_elapsed_seconds=60)
            )
        )
        unlimited = await store.create_task(
            TaskCreate(type="job", event_id="unlimited", retry_budget=TaskRetryBudget.unlimited())
        )
        now[0] += timedelta(days=10)
        await reconcile_task_event_deadlines(store)
        assert (await store.load_task(bounded.id)).status is TaskStatus.CANCELLED
        assert (await store.load_task(unlimited.id)).status is TaskStatus.PENDING
        assert (
            await store.load_task_retry_key(bounded.invocation.event_retry_key)
        ).given_up_at == now[0]

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_event_deadline_retains_ownership_until_cancellation_cleanup_finishes(
    backend, tmp_path, request
):
    async def scenario():
        store = _store(backend, tmp_path, request)
        app = CayuApp(task_store=store, enable_logging=False)
        # Allow SQLite admission to finish before testing the owned cleanup window.
        budget = TaskRetryBudget(max_attempts=None, max_elapsed_seconds=1.0)
        task = await app.create_task(TaskCreate(type="job", event_id="one", retry_budget=budget))
        cancelled, release = asyncio.Event(), asyncio.Event()

        async def handler(app, task, worker_id):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                # An owned operation is still draining; cancellation is not proof of exit.
                await release.wait()

        worker = asyncio.create_task(
            run_task_worker(app, store, handler, worker_id="owner", max_tasks=1, lease_seconds=1)
        )
        try:
            await asyncio.wait_for(cancelled.wait(), 5)
            state = await store.refresh_task_retry_key(task.invocation.event_retry_key)
            assert state.status is TaskRetryKeyStatus.GIVEN_UP
            assert not worker.done()
            current = await store.load_task(task.id)
            assert current.worker_id == "owner"
            assert current.status is TaskStatus.CLAIMED
            assert await store.reclaim_expired() == []
            with pytest.raises(TaskRetryBudgetExhausted):
                await app.create_task(TaskCreate(type="job", event_id="one", retry_budget=budget))
        finally:
            release.set()
            await asyncio.wait_for(worker, 5)
            await app.aclose()
        assert (await store.load_task(task.id)).status in {TaskStatus.FAILED, TaskStatus.CANCELLED}

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_success_closes_event_and_cancels_other_admitted_work(backend, tmp_path, request):
    from cayu.tasks import TaskEventAlreadyCompleted

    async def scenario():
        store = _store(backend, tmp_path, request)
        app = CayuApp(task_store=store, enable_logging=False)
        try:
            first = await app.create_task(TaskCreate(type="job", event_id="one"))
            second = await app.create_task(TaskCreate(type="job", event_id="one"))
            await store.complete_task(first.id, {"ok": True})
            with pytest.raises(TaskEventAlreadyCompleted):
                await app.create_task(TaskCreate(type="job", event_id="one"))
            await reconcile_task_event_deadlines(store)
            assert (await store.load_task(second.id)).status is TaskStatus.CANCELLED
            state = await store.load_task_retry_key(first.invocation.event_retry_key)
            assert state.status is TaskRetryKeyStatus.SUCCEEDED
            assert state.settled_attempts == 2
        finally:
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_scheduled_event_deadline_cancels_using_schedule_revision(backend, tmp_path, request):
    from cayu.tasks import TaskSchedulePolicy

    now = [datetime(2026, 10, 27, tzinfo=UTC)]

    async def scenario():
        store = _store(backend, tmp_path, request, clock=lambda: now[0])
        app = CayuApp(task_store=store, enable_logging=False)
        try:
            task = await app.create_task(
                TaskCreate(
                    task_id="scheduled",
                    type="job",
                    retry_key="scheduled-event",
                    retry_budget=TaskRetryBudget(max_elapsed_seconds=60),
                    available_at=now[0] + timedelta(minutes=10),
                    schedule_policy=TaskSchedulePolicy(),
                )
            )
            now[0] += timedelta(seconds=61)
            await reconcile_task_event_deadlines(store)
            final = await store.load_task(task.id)
            assert final.status is TaskStatus.CANCELLED
            assert final.schedule.revision == task.schedule.revision + 1
            state = await store.load_task_retry_key("scheduled-event")
            assert state.given_up_reason.value == "elapsed_exhausted"
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_sqlite_restart_preserves_event_deadline_and_terminal_state(tmp_path, request):
    now = [datetime(2026, 10, 27, tzinfo=UTC)]
    budget = TaskRetryBudget(max_elapsed_seconds=60)

    async def scenario():
        store = _store("sqlite", tmp_path, request, clock=lambda: now[0])
        first = await store.create_task(TaskCreate(type="job", event_id="one", retry_budget=budget))
        await store.fail_task(first.id, {"code": "retry"})
        await store.close()
        now[0] += timedelta(seconds=50)
        store = _store("sqlite", tmp_path, request, clock=lambda: now[0])
        second = await store.create_task(
            TaskCreate(type="job", event_id="one", retry_budget=budget)
        )
        assert second.invocation.event_retry_key == first.invocation.event_retry_key
        assert await store.task_event_remaining_seconds(first.invocation.event_retry_key) == 10
        await store.close()
        now[0] += timedelta(seconds=11)
        store = _store("sqlite", tmp_path, request, clock=lambda: now[0])
        await reconcile_task_event_deadlines(store)
        assert (await store.load_task(second.id)).status is TaskStatus.CANCELLED
        await store.close()
        store = _store("sqlite", tmp_path, request, clock=lambda: now[0])
        with pytest.raises(TaskRetryBudgetExhausted):
            await store.create_task(TaskCreate(type="job", event_id="one", retry_budget=budget))
        await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_native_successors_and_resubmissions_share_attempt_capacity(backend, tmp_path, request):
    from cayu.tasks import (
        TaskQuery,
        TaskRetryAttemptDisposition,
        TaskRetryAttemptReport,
        TaskRetryPolicy,
    )

    async def scenario():
        store = _store(backend, tmp_path, request)
        app = CayuApp(task_store=store, enable_logging=False)
        budget = TaskRetryBudget(max_attempts=3)
        policy = TaskRetryPolicy(max_attempts=10)
        try:
            first = await app.create_task(
                TaskCreate(type="job", event_id="one", retry_budget=budget, retry_policy=policy)
            )
            await app.create_task(
                TaskCreate(type="job", event_id="one", retry_budget=budget, retry_policy=policy)
            )

            async def handler(app, task, worker_id):
                return TaskRetryAttemptReport(
                    idempotency_key=f"failure-{task.id}",
                    disposition=TaskRetryAttemptDisposition.RETRYABLE_FAILURE,
                    error={"code": "temporary"},
                )

            assert (
                await asyncio.wait_for(
                    run_task_worker(
                        app, store, handler, worker_id="owner", max_tasks=3, poll_interval_s=0.01
                    ),
                    15,
                )
                == 3
            )
            tasks = await store.list_tasks(TaskQuery(type="job"))
            assert len(tasks) == 3
            assert all(
                task.invocation.event_retry_key == first.invocation.event_retry_key
                for task in tasks
            )
            state = await store.load_task_retry_key(first.invocation.event_retry_key)
            assert state.attempts == state.settled_attempts == 3
            assert state.given_up_reason.value == "attempts_exhausted"
        finally:
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_paused_human_wait_does_not_reset_event_deadline(backend, tmp_path, request):
    now = [datetime(2026, 10, 27, tzinfo=UTC)]

    async def scenario():
        store = _store(backend, tmp_path, request, clock=lambda: now[0])
        task = await store.create_task(
            TaskCreate(
                type="job", event_id="one", retry_budget=TaskRetryBudget(max_elapsed_seconds=60)
            )
        )
        now[0] += timedelta(seconds=50)
        await store.pause_task(task.id, reason="waiting_for_approval")
        now[0] += timedelta(seconds=11)
        await reconcile_task_event_deadlines(store)
        assert (await store.load_task(task.id)).status is TaskStatus.CANCELLED
        assert (
            await store.load_task_retry_key(task.invocation.event_retry_key)
        ).given_up_reason.value == "elapsed_exhausted"

    asyncio.run(scenario())


def test_custom_store_must_support_automatic_event_execution():
    class LegacyStore(InMemoryTaskStore):
        supports_task_event_execution = False

    async def scenario():
        app = CayuApp(task_store=LegacyStore(), enable_logging=False)
        try:
            with pytest.raises(NotImplementedError, match="event execution"):
                await app.create_task(TaskCreate(type="job"))
        finally:
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_draining_attempt_does_not_starve_later_expired_events(backend, tmp_path, request):
    from cayu.tasks.retry_budgets import enforce_task_event_deadline

    now = [datetime(2026, 10, 27, tzinfo=UTC)]

    async def scenario():
        store = _store(backend, tmp_path, request, clock=lambda: now[0])
        budget = TaskRetryBudget(max_elapsed_seconds=60)
        first = await store.create_task(
            TaskCreate(type="job", event_id="first", retry_budget=budget)
        )
        claimed = await store.claim_task("owner")
        assert claimed.id == first.id
        second = await store.create_task(
            TaskCreate(type="job", event_id="second", retry_budget=budget)
        )
        now[0] += timedelta(seconds=61)
        await enforce_task_event_deadline(store, claimed)
        assert (await store.load_task(first.id)).status is TaskStatus.CLAIMED
        assert await store.expired_task_event_ids(limit=1) == [second.id]

    asyncio.run(scenario())


def test_postgres_successor_admission_reads_time_after_event_lock(request):
    import psycopg

    from cayu.storage.migrations import SchemaMode
    from cayu.storage.tasks_postgres import PostgresTaskStore
    from cayu.tasks import TaskRetryAttemptDisposition, TaskRetryPolicy, TaskRetrySettlementRequest

    dsn = request.getfixturevalue("event_postgres_dsn")
    now = [datetime(2026, 10, 27, tzinfo=UTC)]

    async def scenario():
        store = PostgresTaskStore(dsn, schema_mode=SchemaMode.CREATE, clock=lambda: now[0])
        try:
            await store.create_task(
                TaskCreate(
                    type="job",
                    retry_key="locked-successor",
                    retry_policy=TaskRetryPolicy(max_attempts=5),
                    retry_budget=TaskRetryBudget(max_elapsed_seconds=60),
                )
            )
            claimed = await store.claim_task("owner")
            async with await psycopg.AsyncConnection.connect(dsn) as holder:
                await holder.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 449))",
                    ("cayu:verified-work:retry-key:locked-successor",),
                )
                settlement = asyncio.create_task(
                    store.settle_task_retry_attempt(
                        TaskRetrySettlementRequest(
                            task_id=claimed.id,
                            worker_id="owner",
                            lease_expires_at=claimed.lease_expires_at,
                            idempotency_key="settle",
                            causal_budget_id=claimed.retry_series.causal_budget_id,
                            disposition=TaskRetryAttemptDisposition.RETRYABLE_FAILURE,
                            error={"code": "temporary"},
                        )
                    )
                )
                async with await psycopg.AsyncConnection.connect(dsn) as observer:
                    for _ in range(500):
                        cursor = await observer.execute(
                            "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' AND NOT granted"
                        )
                        if (await cursor.fetchone())[0]:
                            break
                        await asyncio.sleep(0.01)
                    else:
                        raise AssertionError("Settlement never waited for the event lock.")
                now[0] += timedelta(seconds=61)
                await holder.rollback()
            receipt = await settlement
            assert receipt.successor is None
            assert receipt.task.retry_series.disposition.value == "elapsed_exhausted"
            state = await store.load_task_retry_key("locked-successor")
            assert state.attempts == 1
            assert state.given_up_at == now[0]
        finally:
            await store.close()

    asyncio.run(scenario())


def test_event_identity_survives_access_policy_revision_but_isolates_subjects():
    from types import SimpleNamespace

    from cayu.tasks.retry_budgets import task_event_key

    request = TaskCreate(type="job", event_id="same-event")

    def key(subject, revision):
        invocation = SimpleNamespace(
            origin=InvocationOrigin(trust="unattributed"),
            access_labels={"tenant": "one"},
            resource_access=SimpleNamespace(
                authority="policy", subject=subject, policy_revision=revision
            ),
        )
        return task_event_key(request, task_id="attempt", invocation=invocation)

    assert key("alice", 1) == key("alice", 2)
    assert key("alice", 1) != key("bob", 1)


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_verified_actor_event_namespaces_do_not_cancel_each_other(backend, tmp_path, request):
    from cayu.sessions.invocation import TaskExecutionSource
    from cayu.tasks.creation import task_create_with_runtime_invocation

    async def scenario():
        store = _store(backend, tmp_path, request)

        def create(subject, tenant):
            return store.create_task(
                task_create_with_runtime_invocation(
                    TaskCreate(type="job", input={"event_id": "invoice-1"}),
                    source=TaskExecutionSource.HTTP_TASK,
                    verified_origin=InvocationOrigin(
                        trust="server_verified", subject=subject, tenant=tenant
                    ),
                )
            )

        try:
            alice = await create("alice", "a")
            bob = await create("bob", "a")
            other_tenant = await create("alice", "b")
            assert len({t.invocation.event_retry_key for t in (alice, bob, other_tenant)}) == 3
            await store.cancel_task(bob.id)
            await reconcile_task_event_deadlines(store)
            assert (await store.load_task(alice.id)).status is TaskStatus.PENDING
            again = await create("alice", "a")
            assert again.invocation.event_retry_key == alice.invocation.event_retry_key
        finally:
            if hasattr(store, "close"):
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_resubmission_inherits_first_budget_when_omitted(backend, tmp_path, request):
    from cayu.tasks import TaskRetryPolicy

    async def scenario():
        store = _store(backend, tmp_path, request)
        try:
            first = await store.create_task(
                TaskCreate(
                    type="job", retry_key="inherited", retry_policy=TaskRetryPolicy(max_attempts=5)
                )
            )
            await store.create_task(TaskCreate(type="job", retry_key="inherited"))
            state = await store.load_task_retry_key(first.invocation.event_retry_key)
            assert state.budget.max_attempts == 5
            assert state.attempts == 2
        finally:
            if hasattr(store, "close"):
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_concurrent_expiry_cancellation_reconciles_stale_snapshot(backend, tmp_path, request):
    from cayu.tasks.retry_budgets import enforce_task_event_deadline

    async def scenario():
        store = _store(backend, tmp_path, request)
        try:
            task = await store.create_task(
                TaskCreate(type="job", retry_budget=TaskRetryBudget(max_elapsed_seconds=0.01))
            )
            await asyncio.sleep(0.02)
            one, two = await asyncio.gather(
                enforce_task_event_deadline(store, task), enforce_task_event_deadline(store, task)
            )
            assert one.status is two.status is TaskStatus.CANCELLED
        finally:
            if hasattr(store, "close"):
                await store.close()

    asyncio.run(scenario())


def test_event_cancellation_does_not_interrupt_delayed_access_cleanup(monkeypatch):
    from contextlib import asynccontextmanager

    import cayu.resource_access as access

    original = access.execution_access

    @asynccontextmanager
    async def delayed(*args, **kwargs):
        await asyncio.sleep(0.05)
        async with original(*args, **kwargs):
            yield

    monkeypatch.setattr(access, "execution_access", delayed)

    async def scenario():
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        task = await app.create_task(
            TaskCreate(type="job", retry_budget=TaskRetryBudget(max_elapsed_seconds=0.15))
        )
        cleaned = False

        async def handler(*args):
            nonlocal cleaned
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0.2)
                cleaned = True

        try:
            await asyncio.wait_for(
                run_task_worker(app, store, handler, worker_id="owner", max_tasks=1), 5
            )
            assert cleaned
            assert (await store.load_task(task.id)).status in {
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
            }
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_legacy_admissions_are_not_fabricated_as_settled_attempts():
    import json
    from types import SimpleNamespace

    from cayu.tasks.retry_budgets import (
        admit_task_retry_key,
        observe_task_retry_key_transition,
        stored_task_retry_key_state,
    )

    now = datetime.now(UTC)
    state = stored_task_retry_key_state(
        json.dumps(
            {
                "retry_key": "legacy",
                "attempts": 2,
                "budget": {"max_attempts": 3, "max_elapsed_seconds": 1800},
                "first_attempt_at": now.isoformat(),
                "latest_attempt_at": now.isoformat(),
                "latest_task_id": "old",
            }
        ),
        "legacy",
    )
    assert state.legacy_attempts == 2 and state.settled_attempts == 0
    state, admitted = admit_task_retry_key(
        state, retry_key="legacy", budget=state.budget, task_id="new", now=now
    )
    assert admitted
    state = observe_task_retry_key_transition(
        state,
        SimpleNamespace(status=TaskStatus.CLAIMED),
        SimpleNamespace(id="new", status=TaskStatus.FAILED),
        now=now,
    )
    assert state.status is TaskRetryKeyStatus.GIVEN_UP
    assert state.settled_attempts == 1 and state.legacy_attempts == 2


def test_event_inspection_cannot_block_heartbeat_past_lease(monkeypatch):
    from time import monotonic

    from cayu.tasks import TaskClaimLost
    from cayu.tasks.worker import _heartbeat_until, _TaskLeaseAuthority

    async def scenario():
        store = InMemoryTaskStore()
        await store.create_task(TaskCreate(type="job"))
        task = await store.claim_task("owner", lease_seconds=1)
        blocked, release = asyncio.Event(), asyncio.Event()
        original = store.refresh_task_retry_key

        async def slow_refresh(key):
            blocked.set()
            await release.wait()
            return await original(key)

        monkeypatch.setattr(store, "refresh_task_retry_key", slow_refresh)
        authority = _TaskLeaseAuthority(
            lease_expires_at=task.lease_expires_at,
            handoff_id=None,
            task_id=task.id,
            worker_id="owner",
            deadline_monotonic=monotonic() + 0.8,
        )
        try:
            with pytest.raises(TaskClaimLost):
                await asyncio.wait_for(
                    _heartbeat_until(
                        store,
                        task.id,
                        "owner",
                        1,
                        asyncio.Event(),
                        lease_authority=authority,
                        claim_deadline_monotonic=monotonic() + 0.8,
                    ),
                    3,
                )
            assert blocked.is_set()
        finally:
            release.set()
            await asyncio.sleep(0)
            if hasattr(store, "close"):
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("inspection", ["slow", "failed", "cancelled"])
def test_event_inspection_precedes_execution_entry(
    backend, inspection, tmp_path, request, monkeypatch
):
    from cayu.tasks.worker import complete_managed_task

    async def scenario():
        store = _store(backend, tmp_path, request)
        app = CayuApp(task_store=store, enable_logging=False)
        task = await app.create_task(TaskCreate(type="job"))
        original = store.task_event_remaining_seconds
        inspected = asyncio.Event()
        ran = []

        async def inspect(key):
            inspected.set()
            if inspection == "failed":
                raise RuntimeError("inspection unavailable")
            if inspection == "cancelled":
                await asyncio.Event().wait()
            await asyncio.sleep(0.8)
            return await original(key)

        async def handler(runtime, claimed, worker_id):
            ran.append(claimed.id)
            await complete_managed_task(store, claimed, worker_id, {"done": True})

        monkeypatch.setattr(store, "task_event_remaining_seconds", inspect)
        try:
            worker = asyncio.create_task(
                run_task_worker(
                    app, store, handler, worker_id="owner", max_tasks=1, lease_seconds=1
                )
            )
            await asyncio.wait_for(inspected.wait(), 5)
            # Reads have not published dispatch evidence, even while the claim is held.
            assert (await store.load_task(task.id)).started_at is None
            if inspection == "cancelled":
                worker.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await worker
            elif inspection == "failed":
                with pytest.raises(RuntimeError, match="inspection unavailable"):
                    await worker
            else:
                assert await asyncio.wait_for(worker, 5) == 1
            if inspection != "slow":
                assert not ran
                assert (await store.load_task(task.id)).started_at is None
                monkeypatch.setattr(store, "task_event_remaining_seconds", original)
                await asyncio.sleep(1.1)
                await store.reclaim_expired()
                assert (
                    await asyncio.wait_for(
                        run_task_worker(app, store, handler, worker_id="replacement", max_tasks=1),
                        5,
                    )
                    == 1
                )
            assert ran == [task.id]
            assert (await store.load_task(task.id)).status is TaskStatus.COMPLETED
        finally:
            await app.aclose()
            if hasattr(store, "close"):
                await store.close()

    asyncio.run(scenario())
