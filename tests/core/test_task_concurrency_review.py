from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest
from tests.core.task_invocation_fixtures import task_backed_session_invocation
from tests.core.test_task_concurrency_keys import backend as backend
from tests.core.test_task_concurrency_keys import stores

from cayu import CayuApp
from cayu._resource_access_errors import ResourceAccessDenied
from cayu.agents import AgentSpec
from cayu.evals.testing import ScriptedModelProvider
from cayu.events import EventType
from cayu.messages import Message
from cayu.providers import ModelStreamEvent
from cayu.sessions._access_rules import SessionAccessRule, SessionAccessScope, SessionAccessSelector
from cayu.sessions.requests import ResumeRequest, RunRequest
from cayu.tasks import TaskCreate, TaskQuery
from cayu.tasks.access import ScopedTaskAccess
from cayu.tasks.handoff import interrupted_task_handoff_request
from cayu.tasks.records import TaskClaimLost


@pytest.mark.store_semantics
def test_scoped_task_creation_cannot_choose_a_store_wide_concurrency_key(backend, tmp_path):
    async def scenario():
        async with stores(backend, tmp_path) as (store, _):
            key = str(uuid4())
            victim = await store.create_task(TaskCreate(type=key, concurrency_key=key))
            rule = SessionAccessRule(
                selectors=(SessionAccessSelector(key="tenant", values=("alice",)),)
            )
            scope = SessionAccessScope(read=(rule,), create=(rule,), modify=(rule,))

            async def resolve():
                return scope

            access = ScopedTaskAccess(store, admitted=scope, resolve=resolve)
            blocked_id = str(uuid4())
            with pytest.raises(ResourceAccessDenied):
                await access.create(
                    TaskCreate(task_id=blocked_id, type=key, concurrency_key=key),
                    labels={"tenant": "alice"},
                )
            assert await store.load_task(blocked_id) is None
            assert (await store.claim_task("victim-worker", TaskQuery(type=key))).id == victim.id

    asyncio.run(scenario())


@pytest.mark.store_semantics
def test_direct_resume_cannot_execute_a_keyed_task_without_its_claim(backend, tmp_path):
    async def scenario():
        async with stores(backend, tmp_path) as (store, other):
            key = str(uuid4())
            await store.create_task(TaskCreate(type=key, concurrency_key=key))
            task = await store.claim_task("owner", TaskQuery(type=key))
            provider = ScriptedModelProvider(
                [
                    ModelStreamEvent.text_delta("still working"),
                    ModelStreamEvent.completed({"finish_reason": "stop", "end_turn": False}),
                ]
            )
            app = CayuApp(task_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="assistant", model="fake"))
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        messages=[Message.text("user", "work")],
                        task_id=task.id,
                        task_worker_id="owner",
                        task_lease_expires_at=task.lease_expires_at,
                        max_steps=1,
                    )
                )
            ]
            interrupted = next(
                event for event in events if event.type is EventType.SESSION_INTERRUPTED
            )
            attached = await store.load_task(task.id)
            await store.release_attached_task_worker(
                task.id,
                "owner",
                lease_expires_at=attached.lease_expires_at,
            )
            peer = await store.create_task(TaskCreate(type=key, concurrency_key=key))
            assert (await other.claim_task("peer", TaskQuery(type=key))).id == peer.id
            with pytest.raises(TaskClaimLost, match="exact continuation claim"):
                _ = [
                    event
                    async for event in app.resume(
                        ResumeRequest(
                            session_id=interrupted.session_id,
                            messages=[Message.text("user", "continue")],
                            max_steps=1,
                        )
                    )
                ]
            assert len(provider.requests) == 1

    asyncio.run(scenario())


@pytest.mark.store_semantics
def test_restarted_task_can_later_handoff_for_an_interrupted_continuation(backend, tmp_path):
    async def scenario():
        async with stores(backend, tmp_path) as (store, _):
            key = str(uuid4())
            await store.create_task(TaskCreate(type=key, restartable=True, concurrency_key=key))
            task = await store.claim_task("before-restart", TaskQuery(type=key))
            restarted = await store.restart_task(
                task.id, task.worker_id, lease_expires_at=task.lease_expires_at
            )
            assert restarted.status_reason is None
            task = await store.claim_task("after-restart", TaskQuery(type=key))
            session_id = str(uuid4())
            attached = await store.attach_task(
                task.id,
                session_id=session_id,
                session_invocation=await task_backed_session_invocation(store, task.id, session_id),
                worker_id=task.worker_id,
                lease_expires_at=task.lease_expires_at,
            )
            await store.release_interrupted_task_worker(
                interrupted_task_handoff_request(attached, session_run_epoch=1)
            )
            claimed = await store.claim_interrupted_task_continuation(
                "continuation",
                TaskQuery(type=key),
                handoff_id=str(uuid4()),
            )
            assert claimed.task is not None and claimed.task.id == task.id
            assert claimed.task.restart_count == 1

    asyncio.run(scenario())


@pytest.mark.store_semantics
@pytest.mark.parametrize("verified", [False, True])
def test_postgres_renewal_waits_for_transient_key_contention(postgres_dsn, tmp_path, verified):
    from tests.core.test_verified_work_contracts import _contract, _digest

    from cayu.tasks.admission import WorkAttemptAdmissionPrepare, WorkAttemptExecutionClaimRequest

    async def scenario():
        async with stores(("postgres", postgres_dsn), tmp_path) as (store, other):
            key = str(uuid4())
            if verified:
                contract = await store.publish_work_contract(_contract(contract_id=str(uuid4())))
                task = await store.create_task(
                    TaskCreate(type=key, concurrency_key=key, work_contract=contract.reference())
                )
                session_id = str(uuid4())
                admission = await store.prepare_work_attempt_admission(
                    WorkAttemptAdmissionPrepare(
                        admission_id=str(uuid4()),
                        claim_id=str(uuid4()),
                        attempt_id=str(uuid4()),
                        task_id=task.id,
                        session_id=session_id,
                        interaction_id=str(uuid4()),
                        worker_id="verified",
                        execution_owner_id=str(uuid4()),
                        generation=1,
                        lease_seconds=30,
                        kind="initial",
                        source_request_sha256=_digest("source"),
                        contract=contract.reference(),
                        session_invocation=await task_backed_session_invocation(
                            store, task.id, session_id
                        ),
                        source_execution_profile_fingerprint=_digest("profile"),
                    )
                )
                claim = admission.claim

                async def renew():
                    renewed = await store.renew_work_attempt_execution_claim(
                        WorkAttemptExecutionClaimRequest(
                            admission_id=admission.admission_id,
                            claim_id=claim.claim_id,
                            worker_id=claim.worker_id,
                            execution_owner_id=claim.execution_owner_id,
                            generation=claim.generation,
                            lease_seconds=60,
                        )
                    )
                    return renewed.claim.lease_expires_at, claim.lease_expires_at
            else:
                await store.create_task(TaskCreate(type=key, concurrency_key=key))
                task = await store.claim_task("ordinary", TaskQuery(type=key), lease_seconds=30)

                async def renew():
                    renewed = await store.heartbeat(
                        task.id,
                        task.worker_id,
                        lease_expires_at=task.lease_expires_at,
                        extend_seconds=60,
                    )
                    return renewed.lease_expires_at, task.lease_expires_at

            entered = asyncio.Event()
            original = store._lock_concurrency_key_for_renewal

            async def mark_attempt(cur, task_id):
                entered.set()
                return await original(cur, task_id)

            store._lock_concurrency_key_for_renewal = mark_attempt
            await other._ensure_ready()
            async with other._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    ("cayu-task-concurrency:" + key,),
                )
                pending = asyncio.create_task(renew())
                try:
                    await asyncio.wait_for(entered.wait(), 10)
                    await asyncio.sleep(0.05)
                    assert not pending.done()
                finally:
                    await conn.commit()
                    renewed, previous = await asyncio.wait_for(pending, 10)
            assert renewed > previous

    asyncio.run(scenario())


@pytest.mark.store_semantics
def test_operator_cancellation_retains_key_until_ordinary_handler_finishes(backend, tmp_path):
    from datetime import UTC, datetime

    from cayu.tasks import TaskStatus, run_task_worker

    async def scenario():
        async with stores(backend, tmp_path) as (store, other):
            key = str(uuid4())
            first = await store.create_task(TaskCreate(type=key, concurrency_key=key))
            await store.create_task(TaskCreate(type=key, concurrency_key=key))
            app = CayuApp(task_store=store, enable_logging=False)
            entered, release = asyncio.Event(), asyncio.Event()

            async def handler(_app, _task, _worker):
                entered.set()
                await release.wait()

            worker = asyncio.create_task(
                run_task_worker(
                    app,
                    store,
                    handler,
                    worker_id="ordinary-drain",
                    query=TaskQuery(type=key),
                    max_tasks=1,
                    lease_seconds=3,
                    poll_interval_s=0.01,
                )
            )
            try:
                await asyncio.wait_for(entered.wait(), 10)
                owned = await store.load_task(first.id)
                await store.request_claimed_task_cancellation(
                    first.id, owned.worker_id, owned.lease_expires_at
                )
                await asyncio.sleep(3.1)
                assert not worker.done()
                assert (await store.load_task(first.id)).lease_expires_at > datetime.now(UTC)
                assert await other.claim_task("peer", TaskQuery(type=key)) is None
            finally:
                release.set()
                await asyncio.wait_for(worker, 10)
            assert (await store.load_task(first.id)).status is TaskStatus.CANCELLED

    asyncio.run(scenario())


def test_shutdown_then_operator_cancel_keeps_retry_key_until_handler_quiesces(backend, tmp_path):
    from contextlib import suppress
    from datetime import UTC, datetime

    from cayu.tasks import TaskRetryPolicy, TaskStatus, run_task_worker
    from cayu.tasks.retry import TaskRetryAttemptDisposition, TaskRetryAttemptReport

    async def scenario():
        async with stores(backend, tmp_path) as (store, other):
            key = str(uuid4())
            first = await store.create_task(
                TaskCreate(
                    type=key,
                    concurrency_key=key,
                    retry_policy=TaskRetryPolicy(max_attempts=2),
                )
            )
            peer = await store.create_task(TaskCreate(type=key, concurrency_key=key))
            app = CayuApp(task_store=store, enable_logging=False)
            entered, release = asyncio.Event(), asyncio.Event()

            async def handler(_app, _task, _worker):
                entered.set()
                await release.wait()
                return TaskRetryAttemptReport(
                    disposition=TaskRetryAttemptDisposition.SUCCEEDED, result={}
                )

            worker = asyncio.create_task(
                run_task_worker(
                    app,
                    store,
                    handler,
                    worker_id="retry-drain",
                    query=TaskQuery(type=key),
                    max_tasks=1,
                    lease_seconds=3,
                    poll_interval_s=0.01,
                )
            )
            try:
                await asyncio.wait_for(entered.wait(), 10)
                worker.cancel()
                await asyncio.sleep(0)
                await store.cancel_task(first.id)
                await asyncio.sleep(4.2)
                assert not worker.done()
                assert (await store.load_task(first.id)).lease_expires_at > datetime.now(UTC)
                assert await other.claim_task("peer", TaskQuery(type=key)) is None
            finally:
                release.set()
                with suppress(asyncio.CancelledError):
                    await asyncio.wait_for(worker, 10)
            assert (await store.load_task(first.id)).status is TaskStatus.CANCELLED
            assert (await other.claim_task("peer", TaskQuery(type=key))).id == peer.id
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.store_semantics
def test_scoped_pause_cannot_release_key_before_handler_stops(backend, tmp_path):
    from tests.core.test_resource_execution_access import Policy

    from cayu.resource_access import ResourceAccessGrant
    from cayu.tasks import complete_managed_task, run_task_worker
    from cayu.tasks.terminalization import TaskTerminalizationConflict

    async def scenario():
        async with stores(backend, tmp_path) as (store, other):
            key = str(uuid4())
            rule = SessionAccessRule(
                selectors=(SessionAccessSelector(key="tenant", values=("alice",)),)
            )
            scope = SessionAccessScope(
                read=(rule,), create=(rule,), modify=(rule,), execute=(rule,)
            )

            policy = Policy()
            policy.current = ResourceAccessGrant(tasks=scope, sessions=scope)
            app = CayuApp(task_store=store, resource_access_policy=policy, enable_logging=False)
            access = (await app.access("alice")).tasks
            parent = await access.create(TaskCreate(type="parent"), labels={"tenant": "alice"})
            first = await store.create_task(
                TaskCreate(type=key, concurrency_key=key, parent_task_id=parent.id)
            )
            peer = await store.create_task(TaskCreate(type=key, concurrency_key=key))
            entered, release = asyncio.Event(), asyncio.Event()

            async def handler(_app, task, worker_id):
                entered.set()
                await release.wait()
                await complete_managed_task(store, task, worker_id, {})

            worker = asyncio.create_task(
                run_task_worker(
                    app,
                    store,
                    handler,
                    worker_id="paused-owner",
                    query=TaskQuery(type=key),
                    max_tasks=1,
                    poll_interval_s=0.01,
                )
            )
            try:
                await asyncio.wait_for(entered.wait(), 10)
                before = await store.load_task(first.id)
                with pytest.raises(TaskTerminalizationConflict, match="handler stops"):
                    await access.pause(first.id)
                after = await store.load_task(first.id)
                assert after.worker_id == before.worker_id
                assert after.lease_expires_at == before.lease_expires_at
                assert after.status == before.status
                assert not worker.done()
                assert await other.claim_task("peer", TaskQuery(type=key)) is None
            finally:
                release.set()
                await asyncio.wait_for(worker, 10)
            assert (await other.claim_task("peer", TaskQuery(type=key))).id == peer.id

    asyncio.run(scenario())


@pytest.mark.store_semantics
@pytest.mark.parametrize("keyed", [False, True])
def test_verified_admission_remains_authoritative_before_ordinary_hold_checks(
    backend, tmp_path, keyed
):
    from tests.core.test_verified_work_contracts import _contract, _digest
    from tests.core.test_work_attempt_admission import _prepare_request

    from cayu.tasks.admission import WorkAttemptAdmissionActivate, WorkAttemptExecutionClaimLost

    async def scenario():
        async with stores(backend, tmp_path) as (store, other):
            identity = str(uuid4())
            contract = _contract(contract_id=identity)
            await store.publish_work_contract(contract)
            task = await store.create_task(
                TaskCreate(
                    type=identity,
                    concurrency_key=identity if keyed else None,
                    work_contract=contract.reference(),
                )
            )
            request = _prepare_request(
                task_id=task.id,
                session_id=identity,
                session_invocation=await task_backed_session_invocation(store, task.id, identity),
                admission_id=identity,
                claim_id=identity,
                attempt_id=identity,
            ).model_copy(update={"contract": contract.reference()})
            prepared = await store.prepare_work_attempt_admission(request)
            admitted = await store.activate_work_attempt_admission(
                WorkAttemptAdmissionActivate(
                    admission_id=request.admission_id,
                    claim_id=request.claim_id,
                    prepare_request_sha256=prepared.prepare_request_sha256,
                    session_evidence_sha256=_digest("session-evidence"),
                )
            )
            before = await store.load_task(task.id)
            for hold in (store.pause_task, store.block_task, store.mark_task_needs_attention):
                with pytest.raises(WorkAttemptExecutionClaimLost, match="ordinary task holds"):
                    await hold(task.id, reason="ordinary-hold")
                assert await store.load_task(task.id) == before
                assert (
                    await store.load_work_attempt_execution_claim(admitted.claim.claim_id)
                    == admitted.claim
                )
            if keyed:
                await store.create_task(TaskCreate(type=identity, concurrency_key=identity))
                assert await other.claim_task("peer", TaskQuery(type=identity)) is None

    asyncio.run(scenario())
