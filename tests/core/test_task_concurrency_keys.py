"""Durable keyed claims stay in the queue until their shared resource is free."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from uuid import uuid4

import pytest
from pydantic import ValidationError

from cayu.storage import PostgresTaskStore, SQLiteTaskStore
from cayu.storage.migrations import SchemaMode
from cayu.tasks import InMemoryTaskStore, TaskCreate, TaskQuery, TaskStatus
from cayu.tasks.records import TaskClaimLost
from cayu.tasks.terminalization import TaskTerminalizationRequest, TaskTerminalKind


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def backend(request):
    name = request.param
    return name, request.getfixturevalue("postgres_dsn") if name == "postgres" else None


@asynccontextmanager
async def stores(backend, tmp_path):
    name, dsn = backend
    if name == "memory":
        first = InMemoryTaskStore()
        yield first, first
        return
    if name == "sqlite":
        first = SQLiteTaskStore(tmp_path / "tasks.db")
        second = SQLiteTaskStore(tmp_path / "tasks.db")
    else:
        first = PostgresTaskStore(dsn, schema_mode=SchemaMode.CREATE)
        await first.list_tasks(TaskQuery(limit=1))
        second = PostgresTaskStore(dsn, schema_mode=SchemaMode.VALIDATE)
    try:
        yield first, second
    finally:
        await second.close()
        await first.close()


async def settle(store, task, kind=TaskTerminalKind.COMPLETED):
    if kind is TaskTerminalKind.CANCELLED:
        from cayu.tasks._cancellation import _task_cancellation_terminalization_request

        cancelled = await store.request_claimed_task_cancellation(
            task.id, task.worker_id, task.lease_expires_at
        )
        return await store.terminalize_task(
            _task_cancellation_terminalization_request(cancelled, worker_id=task.worker_id)
        )
    return await store.terminalize_task(
        TaskTerminalizationRequest(
            task_id=task.id,
            worker_id=task.worker_id,
            lease_expires_at=task.lease_expires_at,
            handoff_id=task.interrupted_handoff_id,
            idempotency_key=str(uuid4()),
            kind=kind,
            result={} if kind is TaskTerminalKind.COMPLETED else None,
            error={} if kind is not TaskTerminalKind.COMPLETED else None,
        )
    )


@pytest.mark.store_semantics
@pytest.mark.parametrize("ending", ["completed", "failed", "cancelled", "released"])
def test_key_skips_busy_jobs_and_is_reusable_after_settlement(backend, tmp_path, ending):
    async def scenario():
        async with stores(backend, tmp_path) as (first, second):
            kind = str(uuid4())
            key = f"repo/one#{kind}"
            tasks = [
                await first.create_task(TaskCreate(type=kind, concurrency_key=value))
                for value in (key, key, f"repo/two#{kind}", None)
            ]
            query = TaskQuery(type=kind)
            owned = await first.claim_task("owner", query)
            assert owned.id == tasks[0].id
            # Older tasks sharing a key cannot consume a worker slot.
            unrelated = await second.claim_task("other-key", query)
            unkeyed = await second.claim_task("unkeyed", query)
            assert {unrelated.id, unkeyed.id} == {tasks[2].id, tasks[3].id}
            assert await second.claim_task("blocked", query) is None
            assert (await second.load_task(tasks[1].id)).status is TaskStatus.PENDING
            assert (await second.load_task(owned.id)).model_dump()["concurrency_key"] == key
            if ending == "released":
                await first.release_task(
                    owned.id, owned.worker_id, lease_expires_at=owned.lease_expires_at
                )
            else:
                await settle(first, owned, TaskTerminalKind(ending))
            next_task = await second.claim_task("successor", query)
            assert next_task.id == tasks[0 if ending == "released" else 1].id
            assert next_task.concurrency_key == key

    asyncio.run(scenario())


@pytest.mark.store_semantics
def test_expired_owner_cannot_renew_after_same_key_is_reassigned(backend, tmp_path):
    async def scenario():
        async with stores(backend, tmp_path) as (first, second):
            kind = str(uuid4())
            for _ in range(2):
                await first.create_task(TaskCreate(type=kind, concurrency_key=kind))
            query = TaskQuery(type=kind)
            stale = await first.claim_task("lost-process", query, lease_seconds=1)
            # No release or reclaim sweep: disappearance of the owner is sufficient.
            await asyncio.sleep(1.05)
            current = await second.claim_task("replacement", query)
            assert current is not None and current.id != stale.id
            with pytest.raises(TaskClaimLost):
                await first.heartbeat(
                    stale.id, "lost-process", lease_expires_at=stale.lease_expires_at
                )
            # Reclaiming the old row does not admit it alongside the live peer.
            await first.reclaim_expired()
            assert await first.claim_task("third", query) is None
            await settle(second, current)
            reclaimed = await first.claim_task("third", query)
            assert reclaimed.id == stale.id

    asyncio.run(scenario())


@pytest.mark.store_semantics
def test_continuation_and_queue_claims_share_the_same_key(backend, tmp_path):
    from tests.core.task_invocation_fixtures import task_backed_session_invocation

    from cayu.tasks.handoff import interrupted_task_handoff_request

    async def scenario():
        async with stores(backend, tmp_path) as (first, second):
            kind = str(uuid4())
            await first.create_task(TaskCreate(type=kind, concurrency_key=kind))
            query = TaskQuery(type=kind)
            initial = await first.claim_task("original", query)
            session = str(uuid4())
            attached = await first.attach_task(
                initial.id,
                session_id=session,
                session_invocation=await task_backed_session_invocation(first, initial.id, session),
                worker_id=initial.worker_id,
                lease_expires_at=initial.lease_expires_at,
            )
            await first.release_interrupted_task_worker(
                interrupted_task_handoff_request(attached, session_run_epoch=1)
            )
            await first.create_task(TaskCreate(type=kind, concurrency_key=kind))
            owned = await second.claim_task("queued-peer", query)
            assert owned is not None
            blocked = await first.claim_interrupted_task_continuation(
                "continuation", query, handoff_id=str(uuid4())
            )
            assert blocked.task is None
            await settle(second, owned)
            resumed = await first.claim_interrupted_task_continuation(
                "continuation", query, handoff_id=str(uuid4())
            )
            assert resumed.task is not None and resumed.task.id == initial.id
            await second.create_task(TaskCreate(type=kind, concurrency_key=kind))
            assert await second.claim_task("blocked-by-continuation", query) is None
            await settle(first, resumed.task)
            assert await second.claim_task("after-continuation", query) is not None

    asyncio.run(scenario())


@pytest.mark.store_semantics
@pytest.mark.parametrize("isolation", ["read committed", "repeatable read"])
@pytest.mark.parametrize("unrelated", [False, True])
def test_postgres_competing_rows_cannot_commit_two_same_key_leases(
    postgres_dsn, tmp_path, isolation, unrelated
):
    from psycopg.conninfo import conninfo_to_dict, make_conninfo

    options = conninfo_to_dict(postgres_dsn).get("options", "")
    level = isolation.replace(" ", "\\ ")
    isolated_dsn = make_conninfo(
        postgres_dsn, options=f"{options} -c default_transaction_isolation={level}"
    )

    async def scenario():
        async with stores(("postgres", isolated_dsn), tmp_path) as (first, second):
            kind = str(uuid4())
            for _ in range(2):
                await first.create_task(TaskCreate(type=kind, concurrency_key=kind))
            if unrelated:
                await first.create_task(TaskCreate(type=kind, concurrency_key=str(uuid4())))
            committed = asyncio.Event()
            original_key_check = second._claim_concurrency_key

            async def after_competing_commit(cur, task):
                await asyncio.wait_for(committed.wait(), 10)
                return await original_key_check(cur, task)

            second._claim_concurrency_key = after_competing_commit

            async def first_claim():
                try:
                    return await first.claim_task("first", TaskQuery(type=kind))
                finally:
                    committed.set()

            # Hold both distinct row locks before either checks the key. Both
            # candidate queries have a snapshot predating the other's claim.
            ready = asyncio.Event()
            arrivals = 0

            def gated(original):
                async def lock(cur, task):
                    nonlocal arrivals
                    await original(cur, task)
                    arrivals += 1
                    if arrivals == 2:
                        ready.set()
                    await asyncio.wait_for(ready.wait(), 10)

                return lock

            first._lock_local_execution_retry_fence = gated(first._lock_local_execution_retry_fence)
            second._lock_local_execution_retry_fence = gated(
                second._lock_local_execution_retry_fence
            )
            claimed = await asyncio.gather(
                first_claim(),
                second.claim_task("second", TaskQuery(type=kind)),
            )
            assert sum(task is not None for task in claimed) == (2 if unrelated else 1)
            current = await first.list_tasks(TaskQuery(type=kind))
            assert sum(task.status is TaskStatus.CLAIMED for task in current) == (
                2 if unrelated else 1
            )
            assert (
                sum(
                    task.status is TaskStatus.CLAIMED and task.concurrency_key == kind
                    for task in current
                )
                == 1
            )
            assert sum(task.status is TaskStatus.PENDING for task in current) == 1

    asyncio.run(scenario())


@pytest.mark.store_semantics
def test_sqlite_claims_contend_across_independent_threads(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    path = tmp_path / "concurrent.db"

    async def prepare():
        store = SQLiteTaskStore(path)
        try:
            for _ in range(8):
                await store.create_task(TaskCreate(type="same", concurrency_key="repo#1"))
            await store.create_task(TaskCreate(type="other", concurrency_key="repo#2"))
        finally:
            await store.close()

    asyncio.run(prepare())
    barrier = Barrier(4)

    def claim(index):
        async def run():
            store = SQLiteTaskStore(path, schema_mode=SchemaMode.VALIDATE)
            try:
                barrier.wait(timeout=10)
                return await store.claim_task(f"worker-{index}")
            finally:
                await store.close()

        return asyncio.run(run())

    with ThreadPoolExecutor(max_workers=4) as pool:
        claimed = [task for task in pool.map(claim, range(4)) if task is not None]
    assert sorted(task.concurrency_key for task in claimed) == ["repo#1", "repo#2"]


@pytest.mark.parametrize("key", ["", " ", "x y", "x\n", "é", "a" * 257, 123, True])
def test_key_requires_a_bounded_portable_identifier(key):
    with pytest.raises(ValidationError):
        TaskCreate(type="job", concurrency_key=key)


@pytest.mark.store_semantics
def test_keyed_jobs_cannot_bypass_claims_with_direct_start(backend, tmp_path):
    with pytest.raises(ValueError, match="unattached"):
        TaskCreate(type="job", concurrency_key="repo#1", session_id="direct")

    async def scenario():
        async with stores(backend, tmp_path) as (first, _):
            task = await first.create_task(TaskCreate(type="job", concurrency_key=str(uuid4())))
            with pytest.raises(TaskClaimLost, match="exact worker claim"):
                await first.start_task(task.id)
            assert (await first.load_task(task.id)).status is TaskStatus.PENDING

    asyncio.run(scenario())


@pytest.mark.store_semantics
def test_retry_successor_retains_key_and_frozen_attempt_authority(backend, tmp_path):
    from cayu.tasks.records import Task, TaskRetryPolicy
    from cayu.tasks.retry import TaskRetryAttemptDisposition, TaskRetrySettlementRequest

    async def scenario():
        async with stores(backend, tmp_path) as (first, second):
            key = str(uuid4())
            task = await first.create_task(
                TaskCreate(
                    type=key,
                    concurrency_key=key,
                    retry_policy=TaskRetryPolicy(max_attempts=2, initial_backoff_seconds=0),
                )
            )
            claimed = await first.claim_task("retry-owner", TaskQuery(type=key))
            receipt = await first.settle_task_retry_attempt(
                TaskRetrySettlementRequest(
                    task_id=claimed.id,
                    worker_id=claimed.worker_id,
                    lease_expires_at=claimed.lease_expires_at,
                    idempotency_key=str(uuid4()),
                    causal_budget_id=task.retry_series.causal_budget_id,
                    disposition=TaskRetryAttemptDisposition.RETRYABLE_FAILURE,
                    error={"error": "temporary"},
                )
            )
            assert receipt.successor.concurrency_key == key
            assert (await second.load_task(receipt.successor.id)).concurrency_key == key
            changed = receipt.successor.model_dump()
            changed["concurrency_key"] = "different-resource"
            with pytest.raises(ValueError, match="authority"):
                Task.model_validate(changed)
            next_task = await second.claim_task("next-owner", TaskQuery(type=key))
            assert next_task.id == receipt.successor.id
            await first.create_task(TaskCreate(type=key, concurrency_key=key))
            assert await first.claim_task("blocked", TaskQuery(type=key)) is None

    asyncio.run(scenario())


@pytest.mark.store_semantics
def test_verified_admission_and_recovery_cannot_bypass_the_key(backend, tmp_path):
    from tests.core.task_invocation_fixtures import task_backed_session_invocation
    from tests.core.test_verified_work_contracts import _contract, _digest

    from cayu.tasks.admission import (
        WorkAttemptAdmissionPrepare,
        WorkAttemptExecutionClaimLost,
        WorkAttemptExecutionClaimRequest,
    )

    async def scenario():
        async with stores(backend, tmp_path) as (first, second):
            key = str(uuid4())
            query = TaskQuery(type=key)
            await first.create_task(TaskCreate(type=key, concurrency_key=key))
            busy = await first.claim_task("ordinary-owner", query)
            contract = await first.publish_work_contract(_contract(contract_id=str(uuid4())))
            task = await first.create_task(
                TaskCreate(type=key, concurrency_key=key, work_contract=contract.reference())
            )
            session_id = str(uuid4())
            request = WorkAttemptAdmissionPrepare(
                admission_id=str(uuid4()),
                claim_id=str(uuid4()),
                attempt_id=str(uuid4()),
                task_id=task.id,
                session_id=session_id,
                interaction_id=str(uuid4()),
                worker_id="verified-owner",
                execution_owner_id=str(uuid4()),
                generation=1,
                lease_seconds=1,
                kind="initial",
                source_request_sha256=_digest("source"),
                contract=contract.reference(),
                session_invocation=await task_backed_session_invocation(first, task.id, session_id),
                source_execution_profile_fingerprint=_digest("profile"),
            )
            with pytest.raises(WorkAttemptExecutionClaimLost, match="concurrency key"):
                await second.prepare_work_attempt_admission(request)
            await settle(first, busy)
            admission = await second.prepare_work_attempt_admission(request)
            await asyncio.sleep(1.05)
            await first.create_task(TaskCreate(type=key, concurrency_key=key))
            busy = await first.claim_task("after-expiry", query)
            recovery = WorkAttemptExecutionClaimRequest(
                admission_id=admission.admission_id,
                claim_id=str(uuid4()),
                worker_id="recovery-owner",
                execution_owner_id=str(uuid4()),
                generation=2,
                lease_seconds=300,
            )
            with pytest.raises(WorkAttemptExecutionClaimLost, match="concurrency key"):
                await second.claim_work_attempt_recovery(recovery)
            await settle(first, busy)
            recovered = await second.claim_work_attempt_recovery(recovery)
            assert recovered.claim.generation == 2
            await first.create_task(TaskCreate(type=key, concurrency_key=key))
            assert await first.claim_task("blocked-by-recovery", query) is None

    asyncio.run(scenario())


@pytest.mark.store_semantics
def test_postgres_renewal_holds_key_until_its_transaction_commits(postgres_dsn, tmp_path):
    from datetime import UTC, datetime

    async def scenario():
        async with stores(("postgres", postgres_dsn), tmp_path) as (first, second):
            key = str(uuid4())
            for _ in range(2):
                await first.create_task(TaskCreate(type=key, concurrency_key=key))
            query = TaskQuery(type=key)
            owned = await first.claim_task("owner", query, lease_seconds=2)
            entered, release = asyncio.Event(), asyncio.Event()
            database_now = first._database_now

            async def delayed_clock(cur):
                observed = await database_now(cur)
                entered.set()
                await release.wait()
                return observed

            first._database_now = delayed_clock
            renewal = asyncio.create_task(
                first.heartbeat(
                    owned.id,
                    owned.worker_id,
                    lease_expires_at=owned.lease_expires_at,
                    extend_seconds=300,
                )
            )
            try:
                await asyncio.wait_for(entered.wait(), 10)
                # A contender sees the old, now expired committed lease while
                # renewal still owns its transaction. It must not claim a peer.
                delay = (owned.lease_expires_at - datetime.now(UTC)).total_seconds()
                await asyncio.sleep(max(0, delay) + 0.05)
                assert await second.claim_task("contender", query) is None
            finally:
                release.set()
                renewed = await renewal
            assert renewed.lease_expires_at > owned.lease_expires_at
            assert await second.claim_task("after-renewal", query) is None

    asyncio.run(scenario())


@pytest.mark.store_semantics
@pytest.mark.parametrize("expiry", [False, True])
def test_busy_key_does_not_prevent_pending_schedule_nonexecution(backend, tmp_path, expiry):
    from datetime import UTC, datetime, timedelta

    from cayu.tasks.scheduling import TaskMisfirePolicy, TaskScheduleEventType, TaskSchedulePolicy

    async def scenario():
        async with stores(backend, tmp_path) as (first, second):
            now = datetime.now(UTC)
            clock = [now]
            first._clock = second._clock = lambda: clock[0]
            if backend[0] == "postgres":
                first._clock_is_injected = second._clock_is_injected = True
            kind = str(uuid4())
            owner = await first.create_task(TaskCreate(type=kind, concurrency_key=kind))
            leased = await first.claim_task("owner", TaskQuery(type=kind))
            assert leased.id == owner.id
            due = now + timedelta(seconds=10)
            pending = await first.create_task(
                TaskCreate(
                    task_id=str(uuid4()),
                    type=kind,
                    concurrency_key=kind,
                    available_at=due,
                    schedule_policy=TaskSchedulePolicy(
                        expires_at=due + timedelta(seconds=1) if expiry else None,
                        misfire_policy=TaskMisfirePolicy.FIRE_ONCE
                        if expiry
                        else TaskMisfirePolicy.SKIP,
                        misfire_grace_seconds=0,
                    ),
                )
            )
            clock[0] = due + timedelta(seconds=2)
            assert await second.claim_task("poller", TaskQuery(type=kind)) is None
            current = await second.load_task(pending.id)
            assert current.status is TaskStatus.CANCELLED
            assert current.worker_id is None and current.schedule.admitted_at is None
            assert (await second.list_task_schedule_events(pending.id))[-1].type is (
                TaskScheduleEventType.EXPIRED if expiry else TaskScheduleEventType.SKIPPED
            )
            assert (await first.load_task(owner.id)).lease_expires_at == leased.lease_expires_at

    asyncio.run(scenario())
