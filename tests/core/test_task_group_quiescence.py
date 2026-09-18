"""Public task-group barrier conformance, including atomic failure boundaries."""

from __future__ import annotations

import asyncio
import contextlib
import threading
from uuid import uuid4

import pytest
from tests.core.test_task_groups import anyio_backend as anyio_backend

from cayu import (
    CayuApp,
    TaskCreate,
    TaskGraphCreate,
    TaskGraphNode,
    TaskGroupCreate,
    TaskGroupEventType,
    TaskGroupFinalizerStatus,
    TaskGroupPolicy,
    TaskGroupQuiescencePolicy,
    TaskGroupQuiescenceStatus,
)
from cayu.storage.migrations import SchemaMode
from cayu.storage.postgres import PostgresTaskStore
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks.base import InMemoryTaskStore, TaskQuery, TaskStatus
from cayu.tasks.groups import TaskGroupConflict, TaskGroupQuiescenceResolution

pytestmark = pytest.mark.anyio


@pytest.fixture(params=["memory", "sqlite", "postgres"])
async def store(request, tmp_path):
    if request.param == "memory":
        yield InMemoryTaskStore()
        return
    if request.param == "sqlite":
        value = SQLiteTaskStore(tmp_path / "barriers.sqlite")
        try:
            yield value
        finally:
            await value.close()
        return
    # Each scenario uses fixed human-readable identities. Give it its own
    # disposable database rather than accidentally replaying a preceding case.
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    server = request.getfixturevalue("_postgres_server_dsn")
    database = "cayu_barrier_" + uuid4().hex
    connection = await psycopg.AsyncConnection.connect(server, autocommit=True)
    await connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
    value = PostgresTaskStore(make_conninfo(server, dbname=database), schema_mode=SchemaMode.CREATE)
    try:
        yield value
    finally:
        await value.close()
        await connection.execute(
            sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(database))
        )
        await connection.close()


def _postgres_address(store: PostgresTaskStore) -> str:
    # ConnectionInfo.dsn intentionally omits passwords. Reopening or creating
    # a peer must retain the fixture's original authenticated configuration.
    address = store._conninfo
    assert isinstance(address, str)
    return address


def request(*, timeout=60, policy="first_success"):
    return TaskGroupCreate(
        group_id="race",
        graph=TaskGraphCreate(
            graph_id="race-graph",
            nodes=tuple(
                TaskGraphNode(task=TaskCreate(task_id=i, type=i))
                for i in (
                    "a",
                    "b",
                    "finalize",
                )
            ),
        ),
        member_task_ids=("a", "b"),
        policy=TaskGroupPolicy(kind=policy),
        quiescence=TaskGroupQuiescencePolicy(timeout_seconds=timeout),
        finalizer_task_id="finalize",
    )


async def test_idle_loser_and_finalizer_release_commit_together(store):
    app = CayuApp(task_store=store, enable_logging=False)
    creation = request()
    receipt = await app.create_task_group(creation)
    finalizer = await store.load_task("finalize")
    assert finalizer.status is TaskStatus.WAITING_GROUP
    assert await store.claim_task("worker", TaskQuery(type="finalize")) is None
    with pytest.raises(ValueError):
        await store.start_task("finalize")
    await store.complete_task("a", {"ok": True})
    snapshot = await app.load_task_group("race")
    assert snapshot.quiescence.status is TaskGroupQuiescenceStatus.QUIESCENT
    assert snapshot.quiescence.finalizer_status is TaskGroupFinalizerStatus.RELEASED
    assert (await store.load_task("b")).status is TaskStatus.CANCELLED
    assert (await store.load_task("finalize")).status is TaskStatus.PENDING
    events = await app.list_task_group_events("race")
    assert sum(e.type is TaskGroupEventType.FINALIZER_RELEASED for e in events) == 1
    assert await app.create_task_group(creation) == receipt
    assert await app.list_task_group_events("race") == events
    await store.complete_task("finalize", {})
    assert (
        await app.load_task_group("race")
    ).quiescence.finalizer_status is TaskGroupFinalizerStatus.SETTLED


async def test_failure_makes_finalizer_explicitly_ineligible(store):
    app = CayuApp(task_store=store, enable_logging=False)
    await app.create_task_group(request(policy="all"))
    await store.fail_task("a", {"code": "test"})
    snapshot = await app.load_task_group("race")
    assert snapshot.quiescence.status is TaskGroupQuiescenceStatus.QUIESCENT
    assert snapshot.quiescence.finalizer_status is TaskGroupFinalizerStatus.INELIGIBLE
    assert (await store.load_task("b")).status is TaskStatus.CANCELLED
    assert (await store.load_task("finalize")).status is TaskStatus.CANCELLED
    assert await store.claim_task("worker", TaskQuery(type="finalize")) is None


@pytest.mark.parametrize("deleted", ["a", "b", "all"])
@pytest.mark.parametrize("outcome", ["success", "failure", "no_finalizer"])
async def test_terminal_reconciliation_survives_permitted_member_deletion(store, deleted, outcome):
    from cayu.tasks.base import TaskSessionClosureClaim
    from cayu.tasks.graphs import TaskGraphConflict

    app = CayuApp(task_store=store, enable_logging=False)
    creation = request(policy="all" if outcome == "failure" else "first_success")
    if outcome == "no_finalizer":
        creation = creation.model_copy(
            update={
                "finalizer_task_id": None,
                "graph": creation.graph.model_copy(update={"nodes": creation.graph.nodes[:2]}),
            }
        )
    deleted_ids = (
        creation.graph.nodes
        if deleted == "all"
        else tuple(node for node in creation.graph.nodes if node.task.task_id == deleted)
    )
    identities = tuple(node.task.task_id for node in deleted_ids)
    session = "retained-session"
    for node in deleted_ids:
        node.task.session_id = session
    receipt = await app.create_task_group(creation)
    with pytest.raises(TaskGraphConflict):
        await store.delete_session_tasks(session, task_ids=identities, policy=None)
    if outcome == "failure":
        await store.fail_task("a", {"code": "test"})
    else:
        await store.complete_task("a", {})
    if outcome == "success":
        # Quiescence alone must not bypass the still-live finalizer or its retention gate.
        released = await app.reconcile_task_group("race")
        assert released.quiescence.finalizer_status is TaskGroupFinalizerStatus.RELEASED
        with pytest.raises(TaskGraphConflict):
            await store.delete_session_tasks(session, task_ids=identities, policy=None)
        await store.complete_task("finalize", {})
    before = await app.load_task_group("race")
    group_events = await app.list_task_group_events("race")
    graph_events = await store.list_task_graph_events("race-graph")
    await store.claim_session_closure(
        TaskSessionClosureClaim(session_id=session, plan_id="c" * 64, task_ids=identities)
    )
    await store.delete_session_tasks(session, task_ids=identities, policy=None)
    for identity in identities:
        assert await store.load_task(identity) is None

    async def assert_replay(owner):
        replay_app = CayuApp(task_store=owner, enable_logging=False)
        for _ in range(2):
            assert await replay_app.reconcile_task_group("race") == before
            # Retained evidence is not permission to accept a new attention resolution.
            with pytest.raises(TaskGroupConflict, match="attention-required"):
                await replay_app.resolve_task_group_quiescence(
                    TaskGroupQuiescenceResolution(
                        group_id="race",
                        request_sha256=receipt.request_sha256,
                        expected_sequence=before.last_sequence,
                        idempotency_key="not-in-attention",
                    )
                )
            assert await replay_app.create_task_group(creation) == receipt
            assert await replay_app.list_task_group_events("race") == group_events
            assert await owner.list_task_graph_events("race-graph") == graph_events
            assert await owner.claim_task("late-finalizer", TaskQuery(type="finalize")) is None

    await assert_replay(store)
    if isinstance(store, SQLiteTaskStore):
        address = store.path
        await store.close()
        reopened = SQLiteTaskStore(address)
    elif isinstance(store, PostgresTaskStore):
        address = _postgres_address(store)
        await store.close()
        reopened = PostgresTaskStore(address)
    else:
        return
    try:
        await assert_replay(reopened)
    finally:
        await reopened.close()


@pytest.mark.parametrize("hold", ["pause_task", "block_task", "mark_task_needs_attention"])
@pytest.mark.parametrize("with_prerequisite", [False, True])
async def test_hold_resume_cannot_bypass_finalizer_gate(
    store, hold, with_prerequisite, monkeypatch
):
    from cayu.tasks import _graphs
    from cayu.tasks.graphs import TaskGraphEventType

    app = CayuApp(task_store=store, enable_logging=False)
    creation = request()
    if with_prerequisite:
        nodes = (
            *creation.graph.nodes[:-1],
            creation.graph.nodes[-1].model_copy(update={"prerequisite_task_ids": ("prepare",)}),
            TaskGraphNode(task=TaskCreate(task_id="prepare", type="prepare")),
        )
        creation = creation.model_copy(
            update={"graph": creation.graph.model_copy(update={"nodes": nodes})}
        )
    receipt = await app.create_task_group(creation)
    # An ordinary root already received readiness at graph admission.
    admitted_history = await store.list_task_graph_events("race-graph")
    await getattr(store, hold)("a")
    await store.resume_task("a")
    assert await store.list_task_graph_events("race-graph") == admitted_history
    await getattr(store, hold)("finalize")
    await store.resume_task("finalize")
    assert (await store.load_task("finalize")).status is TaskStatus.WAITING_GROUP
    held = await getattr(store, hold)("finalize")
    if with_prerequisite:
        await store.complete_task("prepare", {})
    await store.complete_task("a", {})
    assert (await store.load_task("finalize")).status is held.status
    assert await store.claim_task("publisher", TaskQuery(type="finalize")) is None
    before = await store.list_task_graph_events("race-graph")
    assert not any(e.type is TaskGraphEventType.READY and e.task_id == "finalize" for e in before)
    task_before = await store.load_task("finalize")
    group_before = await app.load_task_group("race")
    events_before = await app.list_task_group_events("race")
    construct_event = _graphs.TaskGraphEvent

    def fail_readiness(**kwargs):
        if kwargs.get("type") is TaskGraphEventType.READY and kwargs.get("task_id") == "finalize":
            raise RuntimeError("readiness event preparation failed")
        return construct_event(**kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(_graphs, "TaskGraphEvent", fail_readiness)
        with pytest.raises(RuntimeError, match="readiness event preparation failed"):
            await store.resume_task("finalize")
    assert await store.load_task("finalize") == task_before
    assert await app.load_task_group("race") == group_before
    assert await app.list_task_group_events("race") == events_before
    assert await store.list_task_graph_events("race-graph") == before
    assert await store.claim_task("publisher", TaskQuery(type="finalize")) is None
    await store.resume_task("finalize")
    assert (await store.load_task("finalize")).status is TaskStatus.PENDING
    history = await store.list_task_graph_events("race-graph")
    ready = [e for e in history if e.type is TaskGraphEventType.READY and e.task_id == "finalize"]
    assert len(ready) == 1
    assert ready[0].sequence == before[-1].sequence + 1
    assert ready[0].status is TaskStatus.PENDING
    assert ready[0].prerequisite_task_ids == (("prepare",) if with_prerequisite else ())
    assert history[:-1] == before
    group_history = await app.list_task_group_events("race")

    async def assert_replay(owner):
        replay_app = CayuApp(task_store=owner, enable_logging=False)
        assert await replay_app.create_task_group(creation) == receipt
        await replay_app.reconcile_task_group("race")
        for _ in range(2):
            await getattr(owner, hold)("finalize")
            await owner.resume_task("finalize")
            assert await owner.list_task_graph_events("race-graph") == history
            assert await replay_app.list_task_group_events("race") == group_history

    await assert_replay(store)
    if isinstance(store, SQLiteTaskStore):
        address = store.path
        await store.close()
        reopened = SQLiteTaskStore(address)
    elif isinstance(store, PostgresTaskStore):
        address = _postgres_address(store)
        await store.close()
        reopened = PostgresTaskStore(address)
    else:
        reopened = store
    try:
        await assert_replay(reopened)
        claimed = await reopened.claim_task("publisher", TaskQuery(type="finalize"))
        assert claimed is not None and claimed.id == "finalize"
        assert await reopened.claim_task("other", TaskQuery(type="finalize")) is None
    finally:
        if reopened is not store:
            await reopened.close()


async def retry_group(app, *, policy="first_success"):
    from cayu import TaskRetryPolicy

    creation = request(policy=policy)
    nodes = tuple(
        TaskGraphNode(
            task=TaskCreate(
                task_id=node.task.task_id,
                type=node.task.type,
                retry_policy=TaskRetryPolicy(max_attempts=3, initial_backoff_seconds=0)
                if node.task.task_id == "b"
                else None,
                metadata={
                    "execution_profile_fingerprint": "b" * 64,
                    "effect_fingerprint": "c" * 64,
                },
            )
        )
        for node in creation.graph.nodes
    )
    return await app.create_task_group(
        creation.model_copy(
            update={
                "graph": creation.graph.model_copy(update={"nodes": nodes}),
            }
        )
    )


async def schedule_retry(store, task, key):
    from cayu import TaskRetryAttemptDisposition, TaskRetrySettlementRequest

    return await store.settle_task_retry_attempt(
        TaskRetrySettlementRequest(
            task_id=task.id,
            worker_id=task.worker_id,
            lease_expires_at=task.lease_expires_at,
            causal_budget_id=task.retry_series.causal_budget_id,
            idempotency_key=key,
            disposition=TaskRetryAttemptDisposition.RETRYABLE_FAILURE,
            error={"code": "temporary"},
        )
    )


async def test_group_decision_cancels_existing_retry_descendants_without_counting_them(store):
    app = CayuApp(task_store=store, enable_logging=False)
    await retry_group(app)
    root = await store.claim_task("worker", TaskQuery(type="b"))
    first = await schedule_retry(store, root, "attempt-one")
    child = await store.claim_task("worker", TaskQuery(type="b"))
    assert child.id == first.successor.id
    second = await schedule_retry(store, child, "attempt-two")
    await store.complete_task("a", {})
    snapshot = await app.load_task_group("race")
    assert snapshot.decision.successful_task_ids == ("a",)
    assert tuple(member.task_id for member in snapshot.members) == ("a", "b")
    assert snapshot.quiescence.status is TaskGroupQuiescenceStatus.QUIESCENT
    assert (await store.load_task(second.successor.id)).status is TaskStatus.CANCELLED
    assert await store.claim_task("late", TaskQuery(type="b")) is None
    with pytest.raises(ValueError):
        await store.start_task(second.successor.id)
    assert (await store.load_task("finalize")).status is TaskStatus.PENDING
    before = await app.list_task_group_events("race")
    assert await schedule_retry(store, child, "attempt-two") == second
    assert await app.list_task_group_events("race") == before


async def test_decisive_retry_failure_fences_its_newborn_successor_atomically(store):
    app = CayuApp(task_store=store, enable_logging=False)
    await retry_group(app, policy="all")
    root = await store.claim_task("worker", TaskQuery(type="b"))
    receipt = await schedule_retry(store, root, "failure")
    assert receipt.successor is not None
    assert (await store.load_task(receipt.successor.id)).status is TaskStatus.CANCELLED
    assert await store.claim_task("late", TaskQuery(type="b")) is None
    assert (
        await app.load_task_group("race")
    ).quiescence.status is TaskGroupQuiescenceStatus.QUIESCENT
    assert (await store.load_task("finalize")).status is TaskStatus.CANCELLED


async def test_running_retry_descendant_drains_before_finalizer(store):
    from cayu import TaskRetryAttemptDisposition, TaskRetryAttemptReport
    from cayu.tasks.worker import run_task_worker

    app = CayuApp(task_store=store, enable_logging=False)
    await retry_group(app)
    root = await store.claim_task("first", TaskQuery(type="b"))
    receipt = await schedule_retry(store, root, "retry")
    entered = threading.Event()
    release = threading.Event()

    def external_work():
        entered.set()
        assert release.wait(10)

    async def handler(_app, task, _worker):
        assert task.id == receipt.successor.id
        await asyncio.to_thread(external_work)
        return TaskRetryAttemptReport(
            idempotency_key="handler-result",
            disposition=TaskRetryAttemptDisposition.SUCCEEDED,
            result={},
        )

    worker = asyncio.create_task(
        run_task_worker(
            app,
            store,
            handler,
            worker_id="successor",
            query=TaskQuery(type="b"),
            lease_seconds=3,
            max_tasks=1,
            poll_interval_s=0.01,
        )
    )
    try:
        async with asyncio.timeout(5):
            while not entered.is_set():
                await asyncio.sleep(0.01)
        await store.complete_task("a", {})
        assert (await app.load_task_group("race")).quiescence.unsettled_task_ids == ("b",)
        assert await store.claim_task("finalizer", TaskQuery(type="finalize")) is None
        assert (
            await store.load_task(receipt.successor.id)
        ).status_reason == "retry_cancellation_requested"
        await asyncio.sleep(0.15)
        assert not worker.done()
        release.set()
        assert await asyncio.wait_for(worker, 5) == 1
        snapshot = await app.reconcile_task_group("race")
        assert snapshot.quiescence.status is TaskGroupQuiescenceStatus.QUIESCENT
        assert (await store.load_task(receipt.successor.id)).status is TaskStatus.CANCELLED
        assert await store.claim_task("finalizer", TaskQuery(type="finalize")) is not None
    finally:
        release.set()
        if not worker.done():
            worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker


@pytest.mark.parametrize(
    "failure",
    [
        "ack_loss",
        "cancel",
        "repeated_cancel",
        "settlement_retry",
        "settlement_ack_loss",
        "wrong_claim",
        "missing_claim",
        "queued_heartbeat",
    ],
)
async def test_undispatched_verified_preparation_entry_is_settled(store, failure, monkeypatch):
    from tests.core.test_verified_task_worker import _contract, _RecordingProvider, _StaticHandler

    from cayu import AgentSpec
    from cayu._exception_groups import iter_exception_tree
    from cayu.runtime.verified_task_worker import VerifiedTaskWorker
    from cayu.tasks.contracts import WorkCompletionConflict

    entered, release = asyncio.Event(), asyncio.Event()
    settling, release_settlement = asyncio.Event(), asyncio.Event()
    mark = type(store).mark_claimed_task_execution_started
    settle = type(store)._settle_task_group_execution
    settlement_calls = 0
    repaired = False
    load = type(store).load_task
    readback_release, renewal_waiting = asyncio.Event(), asyncio.Event()
    renewals_after_handoff = []

    if failure == "queued_heartbeat":
        from cayu.runtime.verified_task_worker import _LeaseOwner

        original_heartbeat = _LeaseOwner.heartbeat
        native_heartbeat = type(store).heartbeat

        async def queue_heartbeat(lease_owner):
            if entered.is_set():
                renewal_waiting.set()
            return await original_heartbeat(lease_owner)

        async def observe_renewal(instance, *args, **kwargs):
            if entered.is_set():
                renewals_after_handoff.append(True)
            return await native_heartbeat(instance, *args, **kwargs)

        async def blocked_readback(instance, task_id):
            if task_id == "b" and entered.is_set():
                await readback_release.wait()
            return await load(instance, task_id)

        monkeypatch.setattr(_LeaseOwner, "heartbeat", queue_heartbeat)
        monkeypatch.setattr(type(store), "heartbeat", observe_renewal)
        monkeypatch.setattr(type(store), "load_task", blocked_readback)

    async def conflicting_readback(instance, task_id):
        task = await load(instance, task_id)
        if task_id == "b" and entered.is_set() and not repaired and task is not None:
            if failure == "missing_claim":
                return None
            return task.model_copy(update={"worker_id": "another-worker"})
        return task

    if failure in {"wrong_claim", "missing_claim"}:
        monkeypatch.setattr(type(store), "load_task", conflicting_readback)

    async def lost_entry(instance, *args):
        await mark(instance, *args)
        entered.set()
        await release.wait()
        raise OSError("Preparation entry acknowledgement lost.")

    async def observe_settlement(instance, task):
        nonlocal settlement_calls
        settlement_calls += 1
        settling.set()
        if failure == "repeated_cancel":
            await release_settlement.wait()
        if failure == "settlement_retry" and not repaired:
            raise OSError("Preparation settlement unavailable.")
        if failure == "settlement_ack_loss" and settlement_calls == 1:
            # Expire the claim so settlement also clears the terminal lease.
            await asyncio.sleep(1.1)
            await settle(instance, task)
            raise OSError("Preparation settlement acknowledgement lost.")
        await settle(instance, task)

    monkeypatch.setattr(type(store), "mark_claimed_task_execution_started", lost_entry)
    monkeypatch.setattr(type(store), "_settle_task_group_execution", observe_settlement)
    app = CayuApp(task_store=store, enable_logging=False)
    provider, handler = _RecordingProvider(), _StaticHandler()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
    contract = await store.publish_work_contract(_contract())
    creation = request()
    await app.create_task_group(
        creation.model_copy(
            update={
                "graph": creation.graph.model_copy(
                    update={
                        "nodes": tuple(
                            node.model_copy(
                                update={
                                    "task": node.task.model_copy(
                                        update={"work_contract": contract.reference()}
                                    )
                                }
                            )
                            if node.task.task_id == "b"
                            else node
                            for node in creation.graph.nodes
                        )
                    }
                )
            }
        )
    )
    owner = VerifiedTaskWorker(
        app,
        handler,
        worker_id="preparer",
        query=TaskQuery(type="b"),
        lease_seconds=3 if failure == "queued_heartbeat" else 1,
        callback_timeout_seconds=2.5 if failure == "queued_heartbeat" else 0.8,
    )
    worker = asyncio.create_task(owner.run(max_tasks=1))
    try:
        await asyncio.wait_for(entered.wait(), 10)
        await store.complete_task("a", {})
        assert await store.claim_task("finalizer", TaskQuery(type="finalize")) is None
        if failure in {"cancel", "repeated_cancel"}:
            worker.cancel()
            if failure == "repeated_cancel":
                await asyncio.wait_for(settling.wait(), 10)
                worker.cancel()
                await asyncio.sleep(0)
                assert await store.claim_task("finalizer", TaskQuery(type="finalize")) is None
                release_settlement.set()
            with pytest.raises(asyncio.CancelledError):
                await worker
            assert worker.cancelled()
            assert worker.cancelling() == (2 if failure == "repeated_cancel" else 1)
        else:
            release.set()
            with pytest.raises(BaseException) as caught:
                await worker
            assert any("acknowledgement lost" in str(e) for e in iter_exception_tree(caught.value))
        if failure == "queued_heartbeat":
            assert renewal_waiting.is_set() and not renewals_after_handoff
            assert owner._preparation_settlement is not None
            readback_release.set()
        if failure == "settlement_retry":
            snapshot = await app.load_task_group("race")
            assert snapshot.quiescence.executions[0].settled_at is None
            with pytest.raises(RuntimeError, match="settlement unavailable"):
                await owner.aclose()
            assert await app.load_task_group("race") == snapshot
            repaired = True
        if failure == "settlement_ack_loss":
            with pytest.raises(RuntimeError, match="settlement acknowledgement lost"):
                await owner.aclose()
        if failure in {"wrong_claim", "missing_claim"}:
            with pytest.raises(WorkCompletionConflict, match="exact claim"):
                await owner.aclose()
            assert settlement_calls == 0
            assert await store.claim_task("finalizer", TaskQuery(type="finalize")) is None
            repaired = True
        await owner.aclose()
        assert not handler.preparations and not provider.requests
        assert settlement_calls == (
            3 if failure == "settlement_retry" else 2 if failure == "settlement_ack_loss" else 1
        )
        assert await store.load_latest_work_attempt_admission("b") is None
        await asyncio.sleep(1.1)
        snapshot = await app.reconcile_task_group("race")
        assert snapshot.quiescence.status is TaskGroupQuiescenceStatus.QUIESCENT
        assert (await store.load_task("b")).status is TaskStatus.CANCELLED
        history = await app.list_task_group_events("race")
        assert sum(event.type is TaskGroupEventType.FINALIZER_RELEASED for event in history) == 1
        assert await app.reconcile_task_group("race") == snapshot
        assert await app.list_task_group_events("race") == history
        assert await store.claim_task("finalizer", TaskQuery(type="finalize")) is not None
    finally:
        release.set()
        release_settlement.set()
        readback_release.set()
        repaired = True
        if not worker.done():
            worker.cancel()
        with contextlib.suppress(BaseException):
            await worker
        await owner.aclose()


@pytest.mark.parametrize("hold", ["pause_task", "block_task", "mark_task_needs_attention"])
async def test_resumed_ordinary_callback_gets_fresh_execution_authority(store, monkeypatch, hold):
    from cayu.tasks.base import TaskClaimLost
    from cayu.tasks.worker import run_task_worker

    app = CayuApp(task_store=store, enable_logging=False)
    await app.create_task_group(request())
    entered, release = threading.Event(), threading.Event()
    entries = []
    mark, heartbeat = type(store).mark_claimed_task_execution_started, type(store).heartbeat

    async def record_entry(instance, *args):
        marked = await mark(instance, *args)
        entries.append(marked)
        return marked

    async def lose_second_lease(instance, *args, **kwargs):
        if entered.is_set():
            raise TaskClaimLost("Resumed callback lost its lease.")
        return await heartbeat(instance, *args, **kwargs)

    monkeypatch.setattr(type(store), "mark_claimed_task_execution_started", record_entry)
    monkeypatch.setattr(type(store), "heartbeat", lose_second_lease)

    def blocked_read():
        entered.set()
        assert release.wait(20)

    calls = 0

    async def handler(_app, task, _worker):
        nonlocal calls
        calls += 1
        if calls == 1:
            await getattr(store, hold)(task.id)
        else:
            await asyncio.to_thread(blocked_read)

    async def run(lease_seconds):
        return await run_task_worker(
            app,
            store,
            handler,
            worker_id="same-worker",
            query=TaskQuery(type="b"),
            lease_seconds=lease_seconds,
            max_tasks=1,
        )

    assert await run(30) == 1
    first = (await app.load_task_group("race")).quiescence.executions[0]
    assert first.settled_at is not None
    assert (await store.resume_task("b")).started_at is None
    # A resumed task can be held again without dispatching. The historical
    # settled execution remains evidence, but its cleared marker is not a
    # conflicting execution. Exercise reopened persistent readers as well.
    for next_hold in ("pause_task", "block_task", "mark_task_needs_attention"):
        reopened = (
            SQLiteTaskStore(store.path)
            if isinstance(store, SQLiteTaskStore)
            else PostgresTaskStore(_postgres_address(store))
            if isinstance(store, PostgresTaskStore)
            else store
        )
        try:
            assert (await reopened.load_task_group("race")).quiescence.executions[0] == first
            await getattr(reopened, next_hold)("b")
            assert (await reopened.resume_task("b")).started_at is None
            assert (await reopened.load_task_group("race")).quiescence.executions[0] == first
        finally:
            if reopened is not store:
                await reopened.close()
    worker = asyncio.create_task(run(1))
    try:
        async with asyncio.timeout(10):
            while not entered.is_set():
                await asyncio.sleep(0.01)
        second = (await app.load_task_group("race")).quiescence.executions[0]
        assert second.worker_id == first.worker_id
        assert second.started_at > first.started_at and second.settled_at is None
        with pytest.raises(TaskGroupConflict):
            await store._settle_task_group_execution(entries[0])
        await asyncio.sleep(1.1)
        await store.complete_task("a", {})
        await store.reclaim_expired()
        assert (await app.reconcile_task_group("race")).quiescence.status is (
            TaskGroupQuiescenceStatus.DRAINING
        )
        assert await store.claim_task("finalizer", TaskQuery(type="finalize")) is None
        assert not worker.done()
        release.set()
        with pytest.raises(TaskClaimLost):
            await worker
        assert (await app.reconcile_task_group("race")).quiescence.status is (
            TaskGroupQuiescenceStatus.QUIESCENT
        )
        assert await store.claim_task("finalizer", TaskQuery(type="finalize")) is not None
        assert calls == 2
    finally:
        release.set()
        if not worker.done():
            worker.cancel()
        with contextlib.suppress(asyncio.CancelledError, TaskClaimLost):
            await worker


@pytest.mark.parametrize(
    "failure", ["precommit", "ack_loss", "readback", "hold_then_precommit", "hold_then_ack_loss"]
)
async def test_drained_preparation_retains_settlement_before_election(store, monkeypatch, failure):
    from tests.core.test_verified_task_worker import _contract, _RecordingProvider, _StaticHandler

    from cayu import AgentSpec
    from cayu.runtime.verified_task_worker import VerifiedTaskWorker

    entered, release = threading.Event(), threading.Event()
    drained = threading.Event()
    fail_hold = failure.startswith("hold_then_")

    def blocked_read():
        entered.set()
        assert release.wait(20)
        drained.set()

    class Handler(_StaticHandler):
        calls = 0

        async def prepare(self, context):
            self.calls += 1
            await asyncio.to_thread(blocked_read)
            if fail_hold:
                raise ValueError("Preparation failed after its read settled.")
            return await super().prepare(context)

    app = CayuApp(task_store=store, enable_logging=False)
    provider, handler = _RecordingProvider(), Handler()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
    contract = await store.publish_work_contract(_contract())
    creation = request()
    await app.create_task_group(
        creation.model_copy(
            update={
                "graph": creation.graph.model_copy(
                    update={
                        "nodes": tuple(
                            node.model_copy(
                                update={
                                    "task": node.task.model_copy(
                                        update={"work_contract": contract.reference()}
                                    )
                                }
                            )
                            if node.task.task_id == "b"
                            else node
                            for node in creation.graph.nodes
                        )
                    }
                )
            }
        )
    )
    owner = VerifiedTaskWorker(
        app,
        handler,
        worker_id="preparer",
        query=TaskQuery(type="b"),
        lease_seconds=1,
        callback_timeout_seconds=0.9,
    )
    settle = type(store)._settle_task_group_execution
    requested = owner._group_cancellation_requested
    failed = False
    acknowledgements = []

    async def failing_settlement(instance, task):
        nonlocal failed
        acknowledgements.append(task.model_copy(deep=True))
        if not failed and failure != "readback":
            failed = True
            if failure.endswith("ack_loss"):
                await settle(instance, task)
            raise OSError("Preparation settlement unavailable.")
        return await settle(instance, task)

    async def failing_readback(task_id):
        nonlocal failed
        if failure == "readback" and drained.is_set() and not failed:
            failed = True
            raise OSError("Preparation election readback unavailable.")
        return await requested(task_id)

    async def failing_hold(instance, hold):
        raise OSError("Preparation hold unavailable.")

    monkeypatch.setattr(type(store), "_settle_task_group_execution", failing_settlement)
    monkeypatch.setattr(owner, "_group_cancellation_requested", failing_readback)
    if fail_hold:
        monkeypatch.setattr(type(store), "hold_work_attempt_preparation", failing_hold)
    running = asyncio.create_task(owner.run(max_tasks=1))
    try:
        async with asyncio.timeout(10):
            while not entered.is_set():
                await asyncio.sleep(0.01)
        if not fail_hold:
            running.cancel()
            assert running.cancelling() == 1
            await asyncio.sleep(0)
            assert not running.done()
        release.set()
        if fail_hold:
            with pytest.raises(Exception):
                await asyncio.wait_for(running, 5)
        else:
            try:
                await asyncio.wait_for(running, 5)
            except asyncio.CancelledError as cancellation:
                assert cancellation.__cause__ is not None
            else:
                pytest.fail("Settlement failure replaced caller cancellation")
            assert running.cancelled() and running.cancelling() == 1
        assert failed and drained.is_set()
        retained = owner._preparation_settlement
        assert retained is not None and retained.started is not None
        assert (await app.load_task_group("race")).decision is None
        # The callback has returned, but a later winner may only release the
        # barrier after positive acknowledgement, never merely lease expiry.
        await asyncio.sleep(1.1)
        await store.complete_task("a", {})
        before = await app.reconcile_task_group("race")
        if not failure.endswith("ack_loss"):
            assert before.quiescence.status is TaskGroupQuiescenceStatus.DRAINING
            assert await store.claim_task("finalizer", TaskQuery(type="finalize")) is None
        await owner.aclose()
        assert owner._preparation_settlement is None
        assert acknowledgements and all(item == retained.started for item in acknowledgements)
        assert handler.calls == 1 and not provider.requests
        assert await store.load_latest_work_attempt_admission("b") is None
        assert (await app.reconcile_task_group("race")).quiescence.status is (
            TaskGroupQuiescenceStatus.QUIESCENT
        )
        events = await app.list_task_group_events("race")
        assert sum(event.type is TaskGroupEventType.FINALIZER_RELEASED for event in events) == 1
        await owner.aclose()
        assert await app.list_task_group_events("race") == events
        assert await store.claim_task("finalizer", TaskQuery(type="finalize")) is not None
    finally:
        release.set()
        if not running.done():
            running.cancel()
        await asyncio.gather(running, return_exceptions=True)
        await owner.aclose()


@pytest.mark.parametrize(
    "failure",
    [
        "read",
        "cancel",
        "repeated_cancel",
        "admission_read",
        "admission_cancel",
        "admission_repeated_cancel",
    ],
)
@pytest.mark.parametrize("commit_ack", [False, True])
async def test_successful_preparation_retains_proof_before_admission(
    store, monkeypatch, failure, commit_ack
):
    from tests.core.test_verified_task_worker import _contract, _RecordingProvider, _StaticHandler

    from cayu import AgentSpec
    from cayu.runtime.verified_task_worker import VerifiedTaskWorker, VerifiedTaskWorkerDraining

    class Handler(_StaticHandler):
        calls = 0

        async def prepare(self, context):
            self.calls += 1
            return await super().prepare(context)

    app = CayuApp(task_store=store, enable_logging=False)
    provider, handler = _RecordingProvider(), Handler()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
    contract = await store.publish_work_contract(_contract())
    creation = request()
    await app.create_task_group(
        creation.model_copy(
            update={
                "graph": creation.graph.model_copy(
                    update={
                        "nodes": tuple(
                            node.model_copy(
                                update={
                                    "task": node.task.model_copy(
                                        update={"work_contract": contract.reference()}
                                    )
                                }
                            )
                            if node.task.task_id == "b"
                            else node
                            for node in creation.graph.nodes
                        )
                    }
                )
            }
        )
    )
    owner = VerifiedTaskWorker(
        app,
        handler,
        worker_id="preparer",
        query=TaskQuery(type="b"),
        lease_seconds=1,
        callback_timeout_seconds=0.9,
    )
    entered, release = asyncio.Event(), asyncio.Event()
    admission_boundary = failure.startswith("admission_")
    failure = failure.removeprefix("admission_")
    observe = type(store)._task_group_cancellation_requested
    load_admission = type(store).load_work_attempt_admission
    settle = type(store)._settle_task_group_execution
    acknowledgements = []
    repaired = False
    admission_reads = []

    async def failing_read(instance, task_id):
        if task_id == "b" and handler.calls:
            entered.set()
            await release.wait()
            raise OSError("Post-preparation election read unavailable.")
        return await observe(instance, task_id)

    async def failing_admission_read(instance, admission_id):
        if handler.calls:
            admission_reads.append(admission_id)
            entered.set()
            await release.wait()
            raise OSError("Post-preparation election read unavailable.")
        return await load_admission(instance, admission_id)

    async def failing_ack(instance, task):
        acknowledgements.append(task.model_copy(deep=True))
        if commit_ack or repaired:
            await settle(instance, task)
        if not repaired:
            raise OSError("Preparation settlement acknowledgement unavailable.")

    if admission_boundary:
        monkeypatch.setattr(type(store), "load_work_attempt_admission", failing_admission_read)
    else:
        monkeypatch.setattr(type(store), "_task_group_cancellation_requested", failing_read)
    monkeypatch.setattr(type(store), "_settle_task_group_execution", failing_ack)
    running = asyncio.create_task(owner.run(max_tasks=1))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert handler.calls == 1 and not provider.requests
        retained = owner._preparation_settlement
        assert retained is not None and retained.started is not None
        assert not acknowledgements
        if failure == "read":
            release.set()
            with pytest.raises(RuntimeError, match="Post-preparation election read unavailable"):
                await asyncio.wait_for(running, 5)
        else:
            running.cancel()
            if failure == "repeated_cancel":
                running.cancel()
            try:
                await asyncio.wait_for(running, 5)
            except asyncio.CancelledError:
                pass
            else:
                pytest.fail("Post-preparation cancellation was not propagated")
            assert running.cancelled()
            assert running.cancelling() == (2 if failure == "repeated_cancel" else 1)
            if admission_boundary:
                assert retained.admission is not None
                with pytest.raises(VerifiedTaskWorkerDraining):
                    await owner.aclose()
                assert not retained.admission.operation.done()
                assert len(admission_reads) == 1
                # Keep reconciliation blocked until we have observed the
                # unsettled barrier. aclose() retains its background operation
                # after timing out; releasing here could commit settlement
                # before the DRAINING assertion below.
        assert owner._preparation_settlement is retained
        assert await store.load_latest_work_attempt_admission("b") is None
        await asyncio.sleep(1.1)
        await store.complete_task("a", {})
        assert (await app.reconcile_task_group("race")).quiescence.status is (
            TaskGroupQuiescenceStatus.DRAINING
        )
        assert await store.claim_task("finalizer", TaskQuery(type="finalize")) is None
        release.set()
        with pytest.raises(RuntimeError, match="settlement acknowledgement unavailable"):
            await owner.aclose()
        assert owner._preparation_settlement is retained
        assert (await app.reconcile_task_group("race")).quiescence.status is (
            TaskGroupQuiescenceStatus.QUIESCENT
            if commit_ack
            else TaskGroupQuiescenceStatus.DRAINING
        )
        if not commit_ack:
            assert await store.claim_task("finalizer", TaskQuery(type="finalize")) is None
        repaired = True
        await owner.aclose()
        assert owner._preparation_settlement is None
        assert len(acknowledgements) == 2
        assert all(item == retained.started for item in acknowledgements)
        assert handler.calls == 1 and not provider.requests
        assert await store.load_latest_work_attempt_admission("b") is None
        assert (await app.reconcile_task_group("race")).quiescence.status is (
            TaskGroupQuiescenceStatus.QUIESCENT
        )
        events = await app.list_task_group_events("race")
        assert sum(event.type is TaskGroupEventType.FINALIZER_RELEASED for event in events) == 1
        await owner.aclose()
        assert await app.list_task_group_events("race") == events
        assert await store.claim_task("finalizer", TaskQuery(type="finalize")) is not None
    finally:
        repaired = True
        release.set()
        if not running.done():
            running.cancel()
        await asyncio.gather(running, return_exceptions=True)
        await owner.aclose()


@pytest.mark.parametrize(
    "failure",
    ["precommit", "ack_loss", "cancel_precommit", "cancel_ack_loss", "readback", "recovered"]
    + [
        "wrong_" + field
        for field in (
            "admission_id",
            "task_id",
            "session_id",
            "attempt_id",
            "interaction_id",
            "source_request_sha256",
            "contract",
            "claim_id",
            "worker_id",
            "execution_owner_id",
            "generation",
            "lease_seconds",
            "task_lease_expires_at",
            "prepare_request_sha256",
        )
    ],
)
async def test_preparation_admission_handoff_requires_positive_transfer(
    store, monkeypatch, failure
):
    from datetime import timedelta

    from tests.core.test_verified_task_worker import _contract, _RecordingProvider, _StaticHandler
    from tests.core.test_work_attempt_lifecycle import _lifecycle_clock_now

    from cayu import AgentSpec
    from cayu.runtime.verified_task_worker import VerifiedTaskWorker, VerifiedTaskWorkerDraining
    from cayu.tasks.admission import (
        WorkAttemptExecutionClaimRequest,
        WorkAttemptRecoveryRequired,
        work_attempt_admission_prepare_sha256,
    )

    app = CayuApp(task_store=store, enable_logging=False)
    provider, handler = _RecordingProvider(), _StaticHandler()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
    contract = await store.publish_work_contract(_contract())
    creation = request()
    await app.create_task_group(
        creation.model_copy(
            update={
                "graph": creation.graph.model_copy(
                    update={
                        "nodes": tuple(
                            node.model_copy(
                                update={
                                    "task": node.task.model_copy(
                                        update={"work_contract": contract.reference()}
                                    )
                                }
                            )
                            if node.task.task_id == "b"
                            else node
                            for node in creation.graph.nodes
                        )
                    }
                )
            }
        )
    )
    owner = VerifiedTaskWorker(
        app,
        handler,
        worker_id="preparer",
        query=TaskQuery(type="b"),
        lease_seconds=5 if failure == "recovered" else 300,
        callback_timeout_seconds=1,
    )
    entered, release = asyncio.Event(), asyncio.Event()
    cancelled = failure.startswith("cancel_")
    commits = "precommit" not in failure
    repaired = False
    calls, acknowledgements = [], []
    prepare = type(store).prepare_work_attempt_admission
    latest = type(store).load_latest_work_attempt_admission
    settle = type(store)._settle_task_group_execution

    async def fail_publication(instance, request):
        calls.append(request.model_copy(deep=True))
        if commits:
            await prepare(instance, request)
        entered.set()
        await release.wait()
        raise OSError("Admission preparation acknowledgement unavailable.")

    async def observe_ack(instance, task):
        acknowledgements.append(task.model_copy(deep=True))
        return await settle(instance, task)

    async def altered_readback(instance, task_id):
        value = await latest(instance, task_id)
        if repaired or not entered.is_set():
            return value
        if failure == "readback":
            raise OSError("Admission handoff readback unavailable.")
        if failure.startswith("wrong_"):
            assert value is not None
            field = failure.removeprefix("wrong_")
            if field == "contract":
                return value.model_copy(
                    update={
                        "contract": value.contract.model_copy(
                            update={"version": value.contract.version + 1}
                        )
                    }
                )
            if field in {
                "claim_id",
                "worker_id",
                "execution_owner_id",
                "generation",
                "lease_seconds",
                "task_lease_expires_at",
            }:
                original = calls[0]
                if field in {"generation", "lease_seconds"}:
                    replacement = getattr(original, field) + 1
                elif field == "task_lease_expires_at":
                    replacement = original.task_lease_expires_at + timedelta(seconds=1)
                else:
                    replacement = "other-owner"
                return value.model_copy(
                    update={
                        "prepare_request_sha256": work_attempt_admission_prepare_sha256(
                            original.model_copy(update={field: replacement})
                        )
                    }
                )
            return value.model_copy(
                update={field: "f" * 64 if field.endswith("sha256") else "other-identity"}
            )
        return value

    monkeypatch.setattr(type(store), "prepare_work_attempt_admission", fail_publication)
    monkeypatch.setattr(type(store), "load_latest_work_attempt_admission", altered_readback)
    monkeypatch.setattr(type(store), "_settle_task_group_execution", observe_ack)
    running = asyncio.create_task(owner.run(max_tasks=1))
    try:
        await asyncio.wait_for(entered.wait(), 10)
        retained = owner._preparation_settlement
        assert retained is not None and retained.admission is not None
        if cancelled:
            running.cancel()
            running.cancel()
            try:
                await asyncio.wait_for(running, 5)
            except asyncio.CancelledError:
                pass
            else:
                pytest.fail("Admission handoff lost caller cancellation")
            assert running.cancelled() and running.cancelling() == 2
            with pytest.raises(VerifiedTaskWorkerDraining):
                await owner.aclose()
            assert not retained.admission.operation.done()
            assert len(calls) == 1 and not acknowledgements
            release.set()
        else:
            release.set()
            with pytest.raises(RuntimeError, match="acknowledgement unavailable"):
                await asyncio.wait_for(running, 10)
        if failure == "recovered":
            original = await latest(store, "b")
            assert original is not None
            async with asyncio.timeout(10):
                while await _lifecycle_clock_now(store) <= original.claim.lease_expires_at:
                    await asyncio.sleep(0.01)
            recovered = await store.claim_work_attempt_recovery(
                WorkAttemptExecutionClaimRequest(
                    admission_id=original.admission_id,
                    claim_id="peer-claim",
                    # Preserve the unsettled group execution identity. This
                    # exercises admission-claim advancement, not permission to
                    # replace a different worker's outstanding execution.
                    worker_id=original.claim.worker_id,
                    execution_owner_id="peer-process",
                    generation=original.claim.generation + 1,
                    lease_seconds=300,
                )
            )
            assert recovered.prepare_request_sha256 == original.prepare_request_sha256
            assert recovered.claim.worker_id == original.claim.worker_id
            assert recovered.claim.claim_id != original.claim.claim_id
            assert recovered.claim.execution_owner_id != original.claim.execution_owner_id
            assert recovered.claim.generation == original.claim.generation + 1
        await store.complete_task("a", {})
        assert await store.claim_task("finalizer", TaskQuery(type="finalize")) is None
        if failure == "readback" or failure.startswith("wrong_"):
            with pytest.raises(Exception):
                await owner.aclose()
            assert owner._preparation_settlement is retained and not acknowledgements
        repaired = True
        if commits:
            with pytest.raises(WorkAttemptRecoveryRequired, match="acquired admission"):
                await owner.aclose()
            assert not acknowledgements
            snapshot = await app.reconcile_task_group("race")
            assert snapshot.quiescence.status is TaskGroupQuiescenceStatus.DRAINING
            execution = next(item for item in snapshot.quiescence.executions if item.task_id == "b")
            assert execution.settled_at is None
        else:
            await owner.aclose()
            assert len(acknowledgements) == 1 and acknowledgements[0] == retained.started
            assert await latest(store, "b") is None
        assert owner._preparation_settlement is None
        assert len(calls) == 1 and len(handler.preparations) == 1 and not provider.requests
        await owner.aclose()
    finally:
        repaired = True
        release.set()
        if not running.done():
            running.cancel()
        await asyncio.gather(running, return_exceptions=True)
        with contextlib.suppress(WorkAttemptRecoveryRequired):
            await owner.aclose()


@pytest.mark.parametrize("repeat_start_clock", [False, True])
async def test_resumed_preparation_cannot_inherit_prior_settlement(
    store, monkeypatch, repeat_start_clock
):
    from tests.core.test_verified_task_worker import _contract, _RecordingProvider, _StaticHandler

    from cayu import AgentSpec
    from cayu.runtime.verified_task_worker import VerifiedTaskWorker
    from cayu.tasks.base import TaskClaimLost

    entered, release = threading.Event(), threading.Event()
    entries = []
    mark, heartbeat = type(store).mark_claimed_task_execution_started, type(store).heartbeat

    async def record_entry(instance, *args):
        with monkeypatch.context() as clock_patch:
            if entries and repeat_start_clock:
                if isinstance(instance, PostgresTaskStore):

                    async def fixed_database_time(cur):
                        return entries[0].started_at

                    clock_patch.setattr(instance, "_database_now", fixed_database_time)
                else:
                    clock_patch.setattr(instance, "_ownership_clock", lambda: entries[0].started_at)
            marked = await mark(instance, *args)
        entries.append(marked)
        return marked

    async def lose_second_lease(instance, *args, **kwargs):
        if entered.is_set():
            raise TaskClaimLost("Second preparation lost its lease.")
        return await heartbeat(instance, *args, **kwargs)

    monkeypatch.setattr(type(store), "mark_claimed_task_execution_started", record_entry)
    monkeypatch.setattr(type(store), "heartbeat", lose_second_lease)

    def blocked_read():
        entered.set()
        assert release.wait(20)

    class Handler(_StaticHandler):
        calls = 0

        async def prepare(self, context):
            self.calls += 1
            if self.calls == 1:
                raise ValueError("Preparation needs operator repair.")
            await asyncio.to_thread(blocked_read)
            return await super().prepare(context)

    app = CayuApp(task_store=store, enable_logging=False)
    provider = _RecordingProvider()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
    contract = await store.publish_work_contract(_contract())
    creation = request()
    await app.create_task_group(
        creation.model_copy(
            update={
                "graph": creation.graph.model_copy(
                    update={
                        "nodes": tuple(
                            node.model_copy(
                                update={
                                    "task": node.task.model_copy(
                                        update={"work_contract": contract.reference()}
                                    )
                                }
                            )
                            if node.task.task_id == "b"
                            else node
                            for node in creation.graph.nodes
                        )
                    }
                )
            }
        )
    )
    handler = Handler()
    owner = VerifiedTaskWorker(
        app,
        handler,
        worker_id="same-preparer",
        query=TaskQuery(type="b"),
        lease_seconds=1,
        callback_timeout_seconds=0.2,
    )
    assert await owner.run(max_tasks=1) == 1
    first = (await app.load_task_group("race")).quiescence.executions[0]
    assert first.settled_at is not None
    assert (await store.resume_task("b")).started_at is None
    worker = asyncio.create_task(owner.run(max_tasks=1))
    try:
        async with asyncio.timeout(10):
            while not entered.is_set():
                await asyncio.sleep(0.01)
        second = (await app.load_task_group("race")).quiescence.executions[0]
        assert second.worker_id == first.worker_id
        assert second.started_at > first.started_at and second.settled_at is None
        with pytest.raises(TaskGroupConflict):
            await store._settle_task_group_execution(entries[0])
        await asyncio.sleep(1.1)
        await store.complete_task("a", {})
        await store.reclaim_expired()
        assert (await app.reconcile_task_group("race")).quiescence.status is (
            TaskGroupQuiescenceStatus.DRAINING
        )
        assert await store.claim_task("finalizer", TaskQuery(type="finalize")) is None
        assert not worker.done() and not provider.requests
        release.set()
        with pytest.raises(TaskClaimLost):
            await worker
        assert (await app.reconcile_task_group("race")).quiescence.status is (
            TaskGroupQuiescenceStatus.QUIESCENT
        )
        assert await store.claim_task("finalizer", TaskQuery(type="finalize")) is not None
        assert handler.calls == 2 and not provider.requests
    finally:
        release.set()
        if not worker.done():
            worker.cancel()
        with contextlib.suppress(asyncio.CancelledError, TaskClaimLost):
            await worker
        await owner.aclose()


@pytest.mark.parametrize(
    "settlement",
    [
        "success",
        "failure",
        "ack_loss",
        "admitted",
        "readback_failure",
        "before_success",
        "before_failure",
        "before_ack_loss",
        "before_readback_failure",
        "before_cancel_failure",
        "before_cancel_ack_loss",
        "before_queued_heartbeat_failure",
    ],
)
async def test_preparation_admission_loses_to_winner_without_stranding_barrier(
    store, monkeypatch, settlement
):
    from tests.core.test_verified_task_worker import _contract, _RecordingProvider, _StaticHandler

    from cayu import AgentSpec
    from cayu.runtime.verified_task_worker import VerifiedTaskWorker
    from cayu.tasks.admission import WorkAttemptAdmissionConflict, WorkAttemptRecoveryRequired

    before_admission = settlement.startswith("before_")
    settlement = settlement.removeprefix("before_")
    cancel_owner = settlement.startswith("cancel_")
    queued_heartbeat = settlement.startswith("queued_heartbeat_")
    settlement = settlement.removeprefix("cancel_").removeprefix("queued_heartbeat_")
    prepared, release = asyncio.Event(), asyncio.Event()
    renewal_waiting = asyncio.Event()

    class Handler(_StaticHandler):
        async def prepare(self, context):
            result = await super().prepare(context)
            if before_admission:
                await store.complete_task("a", {})
            if cancel_owner:
                prepared.set()
                await release.wait()
            return result

    app = CayuApp(task_store=store, enable_logging=False)
    provider, handler = _RecordingProvider(), Handler()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
    contract = await store.publish_work_contract(_contract())
    creation = request()
    await app.create_task_group(
        creation.model_copy(
            update={
                "graph": creation.graph.model_copy(
                    update={
                        "nodes": tuple(
                            node.model_copy(
                                update={
                                    "task": node.task.model_copy(
                                        update={"work_contract": contract.reference()}
                                    )
                                }
                            )
                            if node.task.task_id == "b"
                            else node
                            for node in creation.graph.nodes
                        )
                    }
                )
            }
        )
    )
    admit = app.admit_work_attempt
    hold = type(store).hold_work_attempt_preparation
    failures = 1 if settlement in {"failure", "ack_loss"} else 0
    load_admission = type(store).load_latest_work_attempt_admission
    admission_attempted = False
    readback_failed = False
    hold_requests = []
    renewals_after_handoff = []
    if queued_heartbeat:
        from cayu.runtime.verified_task_worker import _LeaseOwner

        owner_heartbeat = _LeaseOwner.heartbeat
        store_heartbeat = type(store).heartbeat

        async def queued_renewal(lease_owner):
            if hold_requests:
                renewal_waiting.set()
            return await owner_heartbeat(lease_owner)

        async def observe_renewal(instance, *args, **kwargs):
            if hold_requests:
                renewals_after_handoff.append(True)
            return await store_heartbeat(instance, *args, **kwargs)

        monkeypatch.setattr(_LeaseOwner, "heartbeat", queued_renewal)
        monkeypatch.setattr(type(store), "heartbeat", observe_renewal)

    async def win_before_admission(*args, **kwargs):
        nonlocal admission_attempted
        assert not before_admission, "Observed election must prevent admission dispatch"
        if settlement == "admitted":
            await admit(*args, **kwargs)
            await store.complete_task("a", {})
            raise WorkAttemptAdmissionConflict("Admission acknowledgement lost after transfer.")
        await store.complete_task("a", {})
        try:
            return await admit(*args, **kwargs)
        except WorkAttemptAdmissionConflict:
            admission_attempted = True
            raise

    async def fail_readback(instance, task_id):
        nonlocal readback_failed
        if (
            settlement == "readback_failure"
            and (admission_attempted or (before_admission and handler.preparations))
            and not readback_failed
        ):
            readback_failed = True
            raise OSError("Preparation handoff settlement acknowledgement unavailable.")
        return await load_admission(instance, task_id)

    async def fail_hold(instance, request):
        nonlocal failures
        hold_requests.append(request.model_copy(deep=True))
        if failures:
            failures -= 1
            if queued_heartbeat:
                # Let the real periodic heartbeat queue behind the handoff
                # lock before returning the precommit failure.
                await asyncio.wait_for(renewal_waiting.wait(), 5)
            if settlement == "ack_loss":
                await hold(instance, request)
            raise OSError("Preparation handoff settlement acknowledgement unavailable.")
        return await hold(instance, request)

    monkeypatch.setattr(app, "admit_work_attempt", win_before_admission)
    monkeypatch.setattr(type(store), "hold_work_attempt_preparation", fail_hold)
    monkeypatch.setattr(type(store), "load_latest_work_attempt_admission", fail_readback)
    owner = VerifiedTaskWorker(
        app,
        handler,
        worker_id="preparer",
        query=TaskQuery(type="b"),
        lease_seconds=3 if queued_heartbeat else 300,
        callback_timeout_seconds=2.5 if queued_heartbeat else 30,
    )
    running = None
    try:
        if settlement == "admitted":
            with pytest.raises(WorkAttemptRecoveryRequired, match="acquired admission"):
                await owner.run(max_tasks=1)
            await owner.aclose()
            assert await store.load_latest_work_attempt_admission("b") is not None
            assert (await app.reconcile_task_group("race")).quiescence.status is (
                TaskGroupQuiescenceStatus.DRAINING
            )
            assert await store.claim_task("finalizer", TaskQuery(type="finalize")) is None
            assert not provider.requests and len(handler.preparations) == 1
            return
        if cancel_owner:
            running = asyncio.create_task(owner.run(max_tasks=1))
            await asyncio.wait_for(prepared.wait(), 5)
            running.cancel()
            assert running.cancelling() == 1
            await asyncio.sleep(0)
            release.set()
            try:
                await asyncio.wait_for(running, 5)
            except asyncio.CancelledError as cancellation:
                assert cancellation.__cause__ is not None
            else:
                pytest.fail("Preparation settlement replaced caller cancellation")
            assert running.cancelled() and running.cancelling() == 1
            assert owner._preparation_settlement is not None
            if settlement == "failure":
                assert await store.claim_task("finalizer", TaskQuery(type="finalize")) is None
        elif settlement == "success":
            assert await owner.run(max_tasks=1) == 1
        else:
            with pytest.raises(RuntimeError, match="settlement acknowledgement unavailable"):
                await owner.run(max_tasks=1)
            assert owner._preparation_settlement is not None
            if settlement in {"failure", "readback_failure"}:
                assert await store.claim_task("finalizer", TaskQuery(type="finalize")) is None
        await owner.aclose()
        assert owner._preparation_settlement is None
        assert not renewals_after_handoff
        assert hold_requests and all(hold == hold_requests[0] for hold in hold_requests)
        assert len(handler.preparations) == 1 and not provider.requests
        assert await store.load_latest_work_attempt_admission("b") is None
        assert (await store.load_task("b")).status is TaskStatus.CANCELLED
        assert (await app.reconcile_task_group("race")).quiescence.status is (
            TaskGroupQuiescenceStatus.QUIESCENT
        )
        events = await app.list_task_group_events("race")
        assert sum(event.type is TaskGroupEventType.FINALIZER_RELEASED for event in events) == 1
        await owner.aclose()
        assert await app.list_task_group_events("race") == events
        assert await store.claim_task("finalizer", TaskQuery(type="finalize")) is not None
    finally:
        release.set()
        if running is not None:
            if not running.done():
                running.cancel()
            await asyncio.gather(running, return_exceptions=True)
        await owner.aclose()


@pytest.mark.parametrize(
    "phase,control",
    [
        (phase, "return")
        for phase in ("prepare", "propose", "verifier", "resolver", "proposal_commit")
    ]
    + [(phase, control) for phase in ("prepare", "propose") for control in ("cancel", "timeout")]
    + [("prepare", control) for control in ("lease_expiry", "lease_expiry_late_winner")]
    + [("prepare", "hold_failure_late_winner")]
    + [
        ("resolver", control)
        for control in (
            "settlement_failure",
            "settlement_ack_loss",
            "settlement_retry_failure",
            "settlement_retry_child_cancel",
        )
    ]
    + [
        ("resolver_entry", "entry_ack_loss"),
        ("resolver_gate", "return"),
        ("resolver", "cancel"),
        ("resolver", "recovery_race"),
    ],
    # Lease expiry is exercised independently of callback timeout below.
)
async def test_verified_callback_group_stop_does_not_admit_later_work(
    store, phase, control, monkeypatch, caplog, capsys, recwarn
):
    import threading

    from tests.core.test_completion_decision_application import (
        _assert_secret_absent_from_cayu_error,
    )
    from tests.core.test_verified_task_worker import (
        RecordingVerifier,
        _accepted_decision,
        _contract,
        _RecordingProvider,
        _Resolver,
        _StaticHandler,
        _task_result,
    )

    from cayu import AgentSpec, CompletionResultResolutionRequest
    from cayu.runtime.completion_result_resolvers import CompletionResultResolverExecutionError
    from cayu.runtime.verified_task_worker import VerifiedTaskWorker
    from cayu.tasks.base import TaskClaimLost
    from cayu.vaults import SecretRedactor

    entered, release = threading.Event(), threading.Event()
    secret = "group-settlement-retry-secret-canary"
    repeated_settlement = control.startswith("settlement_retry")
    late_winner = control in {"lease_expiry_late_winner", "hold_failure_late_winner"}
    if control == "hold_failure_late_winner":

        async def fail_hold(instance, request):
            raise TaskClaimLost("Preparation hold could not confirm its claim.")

        monkeypatch.setattr(type(store), "hold_work_attempt_preparation", fail_hold)
    if control in {"settlement_failure", "settlement_ack_loss"} or repeated_settlement:
        observe = type(store)._observe_task_group_result_resolution
        failed = False
        failures_left = 3 if repeated_settlement else 1

        async def fail_settlement(instance, *args, settled, **kwargs):
            nonlocal failed, failures_left
            if settled and failures_left:
                failures_left -= 1
                failed = True
                if control == "settlement_ack_loss":
                    await observe(instance, *args, settled=settled, **kwargs)
                if repeated_settlement:
                    if control == "settlement_retry_child_cancel":
                        raise asyncio.CancelledError(secret)
                    raise OSError(secret)
                raise OSError("Resolver settlement publication failed.")
            return await observe(instance, *args, settled=settled, **kwargs)

        monkeypatch.setattr(type(store), "_observe_task_group_result_resolution", fail_settlement)
    if control.startswith("lease_expiry"):
        heartbeat = type(store).heartbeat

        async def lose_heartbeat(instance, *args, **kwargs):
            if entered.is_set():
                raise TaskClaimLost("Preparation heartbeat unavailable.")
            return await heartbeat(instance, *args, **kwargs)

        monkeypatch.setattr(type(store), "heartbeat", lose_heartbeat)

    def blocked_read():
        entered.set()
        assert release.wait(15)

    if phase in {"resolver_entry", "resolver_gate"}:
        observe = type(store)._observe_task_group_result_resolution

        async def pause_resolver_entry(instance, *args, settled, **kwargs):
            if not settled:
                if phase == "resolver_entry":
                    await observe(instance, *args, settled=settled, **kwargs)
                await asyncio.to_thread(blocked_read)
                if phase == "resolver_entry":
                    raise OSError("Resolver entry acknowledgement lost.")
            return await observe(instance, *args, settled=settled, **kwargs)

        monkeypatch.setattr(
            type(store), "_observe_task_group_result_resolution", pause_resolver_entry
        )

    class Handler(_StaticHandler):
        async def prepare(self, context):
            if phase == "prepare":
                await asyncio.to_thread(blocked_read)
                if control == "hold_failure_late_winner":
                    raise ValueError("Preparation failed before admission.")
            return await super().prepare(context)

        async def propose(self, context):
            if phase == "propose":
                await asyncio.to_thread(blocked_read)
            return await super().propose(context)

    class Verifier(RecordingVerifier):
        async def verify(self, request):
            if phase == "verifier":
                await asyncio.to_thread(blocked_read)
            return await super().verify(request)

    class Resolver(_Resolver):
        async def resolve(self, request):
            if phase == "resolver":
                await asyncio.to_thread(blocked_read)
            return await super().resolve(request)

    app = CayuApp(task_store=store, enable_logging=False, secret_redactor=SecretRedactor(secret))
    provider = _RecordingProvider()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
    contract = await store.publish_work_contract(_contract())
    verifier = Verifier(_accepted_decision())
    resolver = Resolver(_task_result())
    app.register_completion_verifier(contract.verifier, verifier)
    app.register_completion_result_resolver(contract.result_resolver, resolver)
    if phase == "proposal_commit":
        submit = type(store).submit_admitted_completion_proposal

        async def pause_after_proposal(owner, request):
            result = await submit(owner, request)
            await asyncio.to_thread(blocked_read)
            return result

        monkeypatch.setattr(
            type(store), "submit_admitted_completion_proposal", pause_after_proposal
        )
    creation = request()
    nodes = tuple(
        node.model_copy(
            update={"task": node.task.model_copy(update={"work_contract": contract.reference()})}
        )
        if node.task.task_id == "b"
        else node
        for node in creation.graph.nodes
    )
    await app.create_task_group(
        creation.model_copy(
            update={
                "graph": creation.graph.model_copy(update={"nodes": nodes}),
            }
        )
    )
    async with VerifiedTaskWorker(
        app,
        Handler(),
        worker_id="verified",
        query=TaskQuery(type="b"),
        lease_seconds=1
        if control.startswith("lease_expiry") or control == "hold_failure_late_winner"
        else 300,
        callback_timeout_seconds=0.2
        if control == "timeout" or control.startswith("lease_expiry")
        else 0.9
        if control == "hold_failure_late_winner"
        else 60,
    ) as owner:
        worker = asyncio.create_task(owner.run(max_tasks=1))
        try:
            async with asyncio.timeout(10):
                while not entered.is_set():
                    await asyncio.sleep(0.01)
            if phase == "resolver" and control == "return":
                snapshot = await app.load_task_group("race")
                execution = next(
                    item for item in snapshot.quiescence.executions if item.task_id == "b"
                )
                marker = execution.result_resolution
                assert marker is not None and marker.settled_at is None
                for decision_id, owner_id in (
                    (marker.decision_id + "-other", marker.owner_id),
                    (marker.decision_id, "f" * 32),
                ):
                    with pytest.raises(TaskGroupConflict):
                        await store._observe_task_group_result_resolution(
                            "b", decision_id, owner_id, settled=True
                        )
                    assert await app.load_task_group("race") == snapshot
            if control != "recovery_race" and not late_winner:
                await store.complete_task("a", {})
            assert await store.claim_task("finalizer", TaskQuery(type="finalize")) is None
            if control.startswith("lease_expiry"):
                await asyncio.sleep(1.1)
                await store.reclaim_expired()
                assert await store.claim_task("replacement", TaskQuery(type="b")) is None
                assert await store.claim_task("finalizer", TaskQuery(type="finalize")) is None
                assert not worker.done()
                retained = await store.load_task("b")
                assert retained.started_at is not None and retained.worker_id == "verified"
            if phase in {"verifier", "resolver", "resolver_entry"}:
                peer_stop = asyncio.Event()
                peer_observed = asyncio.Event()
                winner_committed = asyncio.Event()

                class RecoveryPeer(VerifiedTaskWorker):
                    async def _group_cancellation_requested(self, task_id):
                        observed = await super()._group_cancellation_requested(task_id)
                        if control == "recovery_race":
                            peer_observed.set()
                            await winner_committed.wait()
                        return observed

                    async def _step(self, now, handled):
                        result = await super()._step(now, handled)
                        peer_stop.set()
                        return result

                peer_store = (
                    SQLiteTaskStore(store.path)
                    if isinstance(store, SQLiteTaskStore)
                    else PostgresTaskStore(_postgres_address(store))
                    if isinstance(store, PostgresTaskStore)
                    else store
                )
                peer_app = CayuApp(
                    task_store=peer_store, session_store=app.session_store, enable_logging=False
                )
                peer_app.register_provider(provider, default=True)
                peer_app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
                peer_app.register_completion_verifier(contract.verifier, verifier)
                peer_app.register_completion_result_resolver(contract.result_resolver, resolver)
                try:
                    async with RecoveryPeer(
                        peer_app, Handler(), worker_id="recovery-peer", query=TaskQuery(type="b")
                    ) as peer:
                        pending_peer = asyncio.create_task(peer.run(stop=peer_stop, max_tasks=1))
                        try:
                            if control == "recovery_race":
                                await asyncio.wait_for(peer_observed.wait(), 10)
                                await store.complete_task("a", {})
                                winner_committed.set()
                            assert await pending_peer == 0
                        finally:
                            winner_committed.set()
                            if not pending_peer.done():
                                pending_peer.cancel()
                            with contextlib.suppress(asyncio.CancelledError):
                                await pending_peer
                finally:
                    if peer_store is not store:
                        await peer_store.close()
                assert len(verifier.requests) == (0 if phase == "verifier" else 1)
                assert (await app.reconcile_task_group("race")).quiescence.status is (
                    TaskGroupQuiescenceStatus.DRAINING
                )
            if control == "cancel":
                worker.cancel()
            if control != "return":
                await asyncio.sleep(0.3)
                if phase != "resolver" or control != "cancel":
                    assert not worker.done()
                assert not release.is_set()
                snapshot = await app.reconcile_task_group("race")
                if not late_winner:
                    assert snapshot.quiescence.status is TaskGroupQuiescenceStatus.DRAINING
                else:
                    assert snapshot.decision is None
            if phase != "resolver" or control != "cancel":
                assert not worker.done()
            release.set()
            if (
                control in {"settlement_failure", "settlement_ack_loss", "entry_ack_loss"}
                or repeated_settlement
            ):
                with pytest.raises(CompletionResultResolverExecutionError) as initial_failure:
                    await asyncio.wait_for(worker, 15)
                _assert_secret_absent_from_cayu_error(initial_failure.value, secret)
                if control != "entry_ack_loss":
                    assert failed
                if repeated_settlement:
                    snapshot = await app.load_task_group("race")
                    execution = next(
                        item for item in snapshot.quiescence.executions if item.task_id == "b"
                    )
                    with pytest.raises(Exception) as public_failure:
                        await app.resolve_completion_result(
                            CompletionResultResolutionRequest(
                                task_id="b",
                                decision_id=execution.result_resolution.decision_id,
                                idempotency_key="retry-group-settlement",
                            )
                        )
                    _assert_secret_absent_from_cayu_error(public_failure.value, secret)
                    assert await app.load_task_group("race") == snapshot
                    async with VerifiedTaskWorker(
                        app, Handler(), worker_id="failing-recovery", query=TaskQuery(type="b")
                    ) as peer:
                        retry = asyncio.create_task(peer.run(max_tasks=1))
                        with pytest.raises(Exception) as recovery_failure:
                            await retry
                        assert not retry.cancelled() and retry.cancelling() == 0
                    _assert_secret_absent_from_cayu_error(recovery_failure.value, secret)
                    assert failures_left == 0
                    assert await app.load_task_group("race") == snapshot
                # The original runtime retains the positive callback-return
                # acknowledgement. Recovery retries it without another read.
                async with VerifiedTaskWorker(
                    app, Handler(), worker_id="settlement-recovery", query=TaskQuery(type="b")
                ) as peer:
                    assert await peer.run(max_tasks=1) == 1
            elif control.startswith("lease_expiry") or control == "hold_failure_late_winner":
                with pytest.raises(TaskClaimLost):
                    await asyncio.wait_for(worker, 15)
                if late_winner:
                    if control == "hold_failure_late_winner":
                        await asyncio.sleep(1.1)
                    await store.complete_task("a", {})
                    await app.reconcile_task_group("race")
            elif control == "cancel":
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(asyncio.shield(worker), 15)
                assert worker.cancelled() and worker.cancelling() == 1
                if phase == "resolver":
                    async with asyncio.timeout(10):
                        while True:
                            snapshot = await app.load_task_group("race")
                            execution = next(
                                item
                                for item in snapshot.quiescence.executions
                                if item.task_id == "b"
                            )
                            if execution.result_resolution.settled_at is not None:
                                break
                            await asyncio.sleep(0.01)
                    async with VerifiedTaskWorker(
                        app, Handler(), worker_id="cancel-recovery", query=TaskQuery(type="b")
                    ) as peer:
                        assert await peer.run(max_tasks=1) == 1
            else:
                assert await asyncio.wait_for(worker, 15) == 1
            terminal = await store.load_task("b")
            assert terminal.status is TaskStatus.CANCELLED
            admission = await store.load_latest_work_attempt_admission("b")
            if phase == "prepare":
                assert admission is None
                assert not provider.requests
            else:
                assert admission is not None
                proposal = await store.load_completion_proposal_for_attempt(admission.attempt_id)
                if phase == "propose":
                    assert proposal is None
                else:
                    assert proposal is not None
                    decision = await store.load_completion_decision_for_proposal(
                        proposal.proposal_id
                    )
                    assert (decision is not None) == (
                        phase in {"verifier", "resolver", "resolver_entry", "resolver_gate"}
                    )
            assert len(resolver.requests) == (1 if phase == "resolver" else 0)
            assert len(verifier.requests) == (
                1 if phase in {"verifier", "resolver", "resolver_entry", "resolver_gate"} else 0
            )
            assert (
                await app.load_task_group("race")
            ).quiescence.status is TaskGroupQuiescenceStatus.QUIESCENT
            assert await store.claim_task("finalizer", TaskQuery(type="finalize")) is not None
            captured = capsys.readouterr()
            assert secret not in captured.out + captured.err + caplog.text
            assert all(secret not in str(warning.message) for warning in recwarn)
        finally:
            release.set()
            if not worker.done():
                worker.cancel()
            with contextlib.suppress(
                asyncio.CancelledError, TaskClaimLost, CompletionResultResolverExecutionError
            ):
                await worker


async def test_verified_loser_uses_exact_lifecycle_release_before_finalizer(store):
    from tests.core.test_verified_task_worker import _contract, _RecordingProvider, _StaticHandler

    from cayu import AgentSpec
    from cayu.runtime.verified_task_worker import VerifiedTaskWorker

    entered = asyncio.Event()
    release = asyncio.Event()

    class Provider(_RecordingProvider):
        async def stream(self, request):
            entered.set()
            await release.wait()
            async for event in super().stream(request):
                yield event

    app = CayuApp(task_store=store, enable_logging=False)
    app.register_provider(Provider(), default=True)
    app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
    contract = await store.publish_work_contract(_contract())
    creation = request()
    nodes = tuple(
        TaskGraphNode(
            task=TaskCreate(
                task_id=node.task.task_id,
                type=node.task.type,
                work_contract=contract.reference() if node.task.task_id == "b" else None,
            )
        )
        for node in creation.graph.nodes
    )
    await app.create_task_group(
        creation.model_copy(
            update={
                "graph": creation.graph.model_copy(update={"nodes": nodes}),
            }
        )
    )
    handler = _StaticHandler()
    async with VerifiedTaskWorker(
        app, handler, worker_id="verified", query=TaskQuery(type="b")
    ) as owner:
        worker = asyncio.create_task(owner.run(max_tasks=1))
        try:
            await asyncio.wait_for(entered.wait(), 10)
            await store.complete_task("a", {})
            assert await store._task_group_cancellation_requested("b")
            assert not await store._task_group_cancellation_requested("a")
            assert await store.claim_task("finalizer", TaskQuery(type="finalize")) is None
            release.set()
            assert await asyncio.wait_for(worker, 15) == 1
            assert not handler.proposals
            terminal = await store.load_task("b")
            assert terminal.status is TaskStatus.CANCELLED
            admission = await store.load_latest_work_attempt_admission("b")
            receipt = await store.load_work_attempt_lifecycle_receipt(admission.admission_id)
            assert receipt.task == terminal
            assert receipt.request.kind == "group_cancellation"
            assert receipt.retired_contract_binding
            assert await store.settle_work_attempt_lifecycle(receipt.request) == receipt
            assert (
                await app.load_task_group("race")
            ).quiescence.status is TaskGroupQuiescenceStatus.QUIESCENT
            assert await store.claim_task("finalizer", TaskQuery(type="finalize")) is not None
        finally:
            release.set()
            if not worker.done():
                worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker


async def test_public_recovery_cannot_replace_losing_execution_authority(store, monkeypatch):
    from datetime import timedelta

    from tests.core.test_verified_task_worker import _contract, _RecordingProvider
    from tests.core.test_work_attempt_lifecycle import (
        _advance_lifecycle_clock,
        _lifecycle_clock_now,
    )

    from cayu import AgentSpec, Message, RunRequest
    from cayu.tasks.admission import (
        WorkAttemptAdmissionConflict,
        WorkAttemptExecutionClaimRequest,
        WorkAttemptExecutionRequest,
        WorkAttemptRecoveryRequest,
    )

    clock = [await _lifecycle_clock_now(store)]
    if not isinstance(store, PostgresTaskStore):
        monkeypatch.setattr(store, "_ownership_clock", lambda: clock[0])
    app = CayuApp(task_store=store, enable_logging=False)
    provider = _RecordingProvider()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
    contract = await store.publish_work_contract(_contract())
    creation = request()
    nodes = tuple(
        TaskGraphNode(
            task=TaskCreate(
                task_id=node.task.task_id,
                type=node.task.type,
                work_contract=contract.reference() if node.task.task_id == "b" else None,
            )
        )
        for node in creation.graph.nodes
    )
    await app.create_task_group(
        creation.model_copy(update={"graph": creation.graph.model_copy(update={"nodes": nodes})})
    )
    admitted = await app.admit_work_attempt(
        RunRequest(
            agent_name="worker",
            task_id="b",
            session_id="losing-session",
            messages=[Message.text("user", "Perform the governed work.")],
        ),
        execution=WorkAttemptExecutionRequest(
            admission_id="losing-admission",
            claim_id="original-claim",
            attempt_id="losing-attempt",
            interaction_id="losing-interaction",
            worker_id="original-worker",
            generation=1,
            lease_seconds=5,
        ),
    )
    await store.complete_task("a", {})
    # A group decision must not break exact, non-mutating acknowledgement replay.
    assert (
        await store.claim_work_attempt_recovery(
            WorkAttemptExecutionClaimRequest(
                admission_id=admitted.admission_id,
                claim_id=admitted.claim.claim_id,
                worker_id=admitted.claim.worker_id,
                execution_owner_id=admitted.claim.execution_owner_id,
                generation=1,
                lease_seconds=5,
            )
        )
        == admitted
    )
    await _advance_lifecycle_clock(
        store, clock, admitted.claim.lease_expires_at + timedelta(milliseconds=10)
    )
    task = await store.load_task("b")
    group = await app.load_task_group("race")
    events = await app.list_task_group_events("race")
    session = await app.session_store.load(admitted.session_id)
    recovery = WorkAttemptRecoveryRequest(
        admission_id=admitted.admission_id,
        claim_id="replacement-claim",
        worker_id="replacement-worker",
        generation=2,
        lease_seconds=30,
    )
    for _ in range(2):
        with pytest.raises(WorkAttemptAdmissionConflict, match="Task-group cancellation"):
            await app.recover_work_attempt(recovery)
        assert await store.load_work_attempt_admission(admitted.admission_id) == admitted
        assert (
            await store.load_work_attempt_execution_claim(admitted.claim.claim_id) == admitted.claim
        )
        assert await store.load_work_attempt_execution_claim(recovery.claim_id) is None
        assert await store.load_task("b") == task
        assert await app.load_task_group("race") == group
        assert await app.list_task_group_events("race") == events
        assert await app.session_store.load(admitted.session_id) == session
        assert not provider.requests
    assert await store.claim_task("finalizer", TaskQuery(type="finalize")) is None


@pytest.mark.parametrize(
    "scope", ["ordinary_long_task", "ordinary", "outcome_only", "quiescent", "nonmember"]
)
@pytest.mark.parametrize("session_bytes", [257, 2048])
async def test_group_observation_preserves_public_identifier_contracts(
    store, scope, session_bytes, monkeypatch
):
    from tests.core.test_verified_task_worker import _RecordingProvider

    from cayu import AgentSpec, Message, RunRequest
    from cayu.sessions.base import InMemorySessionStore, SessionStatus
    from cayu.storage.postgres import PostgresSessionStore
    from cayu.storage.sqlite import SQLiteSessionStore

    if isinstance(store, SQLiteTaskStore):
        sessions = SQLiteSessionStore(store.path)
    elif isinstance(store, PostgresTaskStore):
        await store.load_task("a")
        address = _postgres_address(store)
        sessions = PostgresSessionStore(address)
    else:
        sessions = InMemorySessionStore()
    app = CayuApp(task_store=store, session_store=sessions, enable_logging=False)
    provider = _RecordingProvider()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
    task_id = "t" * 257 if scope == "ordinary_long_task" else "a"
    session_id = (
        "s" * session_bytes
        if scope == "ordinary_long_task"
        else "é" * (session_bytes // 2) + "s" * (session_bytes % 2)
    )
    if scope.startswith("ordinary"):
        await app.create_task(TaskCreate(task_id=task_id, type="work"))
    else:
        creation = request()
        if scope == "outcome_only":
            creation = creation.model_copy(update={"quiescence": None, "finalizer_task_id": None})
        elif scope == "nonmember":
            task_id = "independent"
            creation = creation.model_copy(
                update={
                    "graph": creation.graph.model_copy(
                        update={
                            "nodes": (
                                *creation.graph.nodes,
                                TaskGraphNode(task=TaskCreate(task_id=task_id, type="work")),
                            )
                        }
                    )
                }
            )
        await app.create_task_group(creation)
    if scope != "quiescent":

        async def unexpected_observation(_observation):
            raise AssertionError("Nonparticipating work entered group observation")

        monkeypatch.setattr(store, "_observe_task_group_invocation", unexpected_observation)
    try:
        events = [
            event
            async for event in app.run(
                RunRequest(
                    session_id=session_id,
                    task_id=task_id,
                    agent_name="worker",
                    messages=[Message.text("user", "Work")],
                )
            )
        ]
        assert events and provider.requests
        session = await sessions.load(session_id)
        assert session is not None and session.status is SessionStatus.COMPLETED
        task = await store.load_task(task_id)
        assert task is not None and task.status is TaskStatus.COMPLETED
        assert task.session_id == session_id
        if scope == "quiescent":
            snapshot = await app.reconcile_task_group("race")
            execution = next(
                item for item in snapshot.quiescence.executions if item.task_id == task_id
            )
            assert execution.invocation is not None
            assert execution.invocation.session_id == session_id
            assert execution.invocation.owner_settled
            assert execution.invocation.release_record_sha256 is not None
            assert execution.settled_at is not None
            assert snapshot.quiescence.status is TaskGroupQuiescenceStatus.QUIESCENT
            if isinstance(store, (SQLiteTaskStore, PostgresTaskStore)):
                history = await app.list_task_group_events("race")
                await store.close()
                reopened = (
                    SQLiteTaskStore(store.path)
                    if isinstance(store, SQLiteTaskStore)
                    else PostgresTaskStore(address)
                )
                try:
                    restored = CayuApp(task_store=reopened, enable_logging=False)
                    assert await restored.load_task_group("race") == snapshot
                    assert await restored.reconcile_task_group("race") == snapshot
                    assert await restored.list_task_group_events("race") == history
                finally:
                    await reopened.close()
        elif not scope.startswith("ordinary"):
            snapshot = await app.load_task_group("race")
            assert not snapshot.quiescence.executions
    finally:
        if isinstance(sessions, (SQLiteSessionStore, PostgresSessionStore)):
            await sessions.close()


@pytest.mark.parametrize("invalid_session_id", ["é" * 1024 + "s", "", " s", 1, True])
async def test_group_invocation_preserves_session_validation(invalid_session_id):
    from pydantic import ValidationError

    from cayu.tasks.groups import TaskGroupInvocationObligation

    with pytest.raises(ValidationError):
        TaskGroupInvocationObligation(
            task_id="a",
            session_id=invalid_session_id,
            session_instance_id="instance",
            interaction_id="interaction",
            run_epoch=1,
            profile_fingerprint="a" * 64,
        )


@pytest.mark.parametrize(
    "publication", ["normal", "precommit", "ack_loss", "blocked", "deleted_ack_loss"]
)
async def test_ownerless_session_release_advances_group_only_after_runtime_cleanup(
    store, monkeypatch, publication
):
    from tests.core.test_verified_task_worker import _RecordingProvider

    from cayu import (
        AgentSpec,
        InterruptSessionRequest,
        Message,
        RunRequest,
        TaskGroupInvocationSettlementPending,
    )
    from cayu.tasks.base import TaskSessionClosureClaim

    deleted = publication == "deleted_ack_loss"
    if deleted:
        publication = "ack_loss"
    entered, cleaning, cleanup_release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    publication_release = asyncio.Event()
    provider_returned = asyncio.Event()
    repaired = publication == "normal"
    observations = []
    provider_calls = 0
    settlement = None
    publish = type(store)._observe_task_group_invocation

    async def failing_publication(instance, observation):
        if observation.owner_settled and observation.release_record_sha256 is None:
            observations.append(observation.model_copy(deep=True))
            if publication == "blocked":
                await publication_release.wait()
            if not repaired and publication == "precommit":
                raise OSError("Owner-return publication unavailable")
            await publish(instance, observation)
            if not repaired and publication == "ack_loss":
                raise OSError("Owner-return acknowledgement lost")
            return
        await publish(instance, observation)

    monkeypatch.setattr(type(store), "_observe_task_group_invocation", failing_publication)

    class Provider(_RecordingProvider):
        async def stream(self, request):
            nonlocal provider_calls
            provider_calls += 1
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                while not cleanup_release.is_set():
                    try:
                        await cleanup_release.wait()
                    except asyncio.CancelledError:
                        continue  # A provider cleanup operation that cannot abort.
            provider_returned.set()
            async for event in super().stream(request):
                yield event

    app = CayuApp(task_store=store, enable_logging=False)
    app.register_provider(Provider(), default=True)
    app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
    await app.create_task_group(request())

    async def execute():
        return [
            event
            async for event in app.run(
                RunRequest(
                    session_id="group-direct-session",
                    task_id="b",
                    agent_name="worker",
                    messages=[Message.text("user", "Work")],
                )
            )
        ]

    async def interrupt():
        return [
            event
            async for event in app.interrupt_session(
                InterruptSessionRequest(
                    session_id="group-direct-session",
                    reason="task_group_decided",
                )
            )
        ]

    running = asyncio.create_task(execute())
    stopping = None
    try:
        await asyncio.wait_for(entered.wait(), 10)
        await store.complete_task("a", {})
        snapshot = await app.load_task_group("race")
        execution = next(item for item in snapshot.quiescence.executions if item.task_id == "b")
        assert execution.worker_id is None and execution.invocation is not None
        stopping = asyncio.create_task(interrupt())
        await asyncio.wait_for(cleaning.wait(), 10)
        assert await store.claim_task("finalizer", TaskQuery(type="finalize")) is None
        # The runtime may transfer slow cleanup to its retained supervisor.
        # Check actual provider quiescence, not whether its observer returned.
        assert not provider_returned.is_set()
        assert (
            await app.reconcile_task_group("race")
        ).quiescence.status is TaskGroupQuiescenceStatus.DRAINING
        cleanup_release.set()
        if publication == "normal":
            await asyncio.wait_for(asyncio.gather(running, stopping), 20)
        else:
            with pytest.raises(TaskGroupInvocationSettlementPending) as pending:
                await asyncio.wait_for(running, 20)
            settlement = pending.value.settlement
            await asyncio.wait_for(stopping, 20)
            before_retry = await app.reconcile_task_group("race")
            assert before_retry.quiescence.status is (
                TaskGroupQuiescenceStatus.QUIESCENT
                if publication == "ack_loss"
                else TaskGroupQuiescenceStatus.DRAINING
            )
            if publication != "ack_loss":
                assert await store.claim_task("finalizer", TaskQuery(type="finalize")) is None
            if publication == "blocked":
                # A real cancellation of an acknowledgement observer must not
                # cancel or duplicate the already-dispatched store operation.
                retrying = asyncio.create_task(pending.value.settlement.retry())
                await asyncio.sleep(0)
                with pytest.raises(TaskGroupInvocationSettlementPending) as concurrent:
                    await pending.value.settlement.retry()
                assert concurrent.value.settlement is pending.value.settlement
                retrying.cancel()
                with pytest.raises(asyncio.CancelledError) as cancellation:
                    await retrying
                assert retrying.cancelled() and retrying.cancelling() == 1
                assert isinstance(
                    cancellation.value.__cause__, TaskGroupInvocationSettlementPending
                )
                assert cancellation.value.__cause__.settlement is pending.value.settlement
                assert len(observations) == 1
            repaired = True
            publication_release.set()
            if deleted:
                await store.complete_task(before_retry.receipt.finalizer_task_id, {})
                await store.claim_session_closure(
                    TaskSessionClosureClaim(
                        session_id="group-direct-session", plan_id="b" * 64, task_ids=("b",)
                    )
                )
                await store.delete_session_tasks(
                    "group-direct-session", task_ids=("b",), policy=None
                )
                retained = await app.load_task_group("race")
                retained_events = await app.list_task_group_events("race")
            await pending.value.settlement.retry()
            count = len(observations)
            await pending.value.settlement.retry()
            assert len(observations) == count == (1 if publication == "blocked" else 2)
            assert all(item == observations[0] for item in observations)
        snapshot = await app.reconcile_task_group("race")
        assert snapshot.quiescence.status is TaskGroupQuiescenceStatus.QUIESCENT
        assert provider_calls == 1
        if deleted:
            assert await store.load_task("b") is None
            assert snapshot == retained
            assert await app.list_task_group_events("race") == retained_events
        else:
            assert (await store.load_task("b")).status is TaskStatus.CANCELLED
        settled = next(item for item in snapshot.quiescence.executions if item.task_id == "b")
        assert settled.invocation.owner_settled
        events = await app.list_task_group_events("race")
        assert sum(event.type is TaskGroupEventType.FINALIZER_RELEASED for event in events) == 1
        await store._observe_task_group_invocation(
            settled.invocation.model_copy(update={"release_record_sha256": None})
        )
        assert await app.load_task_group("race") == snapshot
        assert await app.list_task_group_events("race") == events
        if deleted:

            async def assert_retained_replay(owner):
                for update in (
                    {"session_id": "other-session"},
                    {"session_instance_id": "other-instance"},
                    {"interaction_id": "other-interaction"},
                    {"run_epoch": settled.invocation.run_epoch + 1},
                    {"profile_fingerprint": "f" * 64},
                    {"owner_settled": False, "release_record_sha256": None},
                    {"release_record_sha256": "f" * 64},
                ):
                    with pytest.raises(TaskGroupConflict, match="retained group owner"):
                        await owner._observe_task_group_invocation(
                            settled.invocation.model_copy(update=update)
                        )
                await owner._observe_task_group_invocation(settled.invocation)
                await owner._observe_task_group_invocation(observations[0])
                assert await owner.load_task_group("race") == snapshot
                assert await owner.list_task_group_events("race") == events
                assert await owner.load_task("b") is None
                assert await owner.claim_task("finalizer", TaskQuery(type="finalize")) is None

            await assert_retained_replay(store)
            reopened = None
            if isinstance(store, SQLiteTaskStore):
                reopened = SQLiteTaskStore(store.path)
            elif isinstance(store, PostgresTaskStore):
                reopened = PostgresTaskStore(_postgres_address(store))
            if reopened is not None:
                try:
                    await assert_retained_replay(reopened)
                finally:
                    await reopened.close()
            assert provider_calls == 1
        else:
            assert await store.claim_task("finalizer", TaskQuery(type="finalize")) is not None
    finally:
        repaired = True
        publication_release.set()
        cleanup_release.set()
        for task in (running, stopping):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(
            *(task for task in (running, stopping) if task is not None), return_exceptions=True
        )
        if settlement is not None:
            await settlement.retry()


@pytest.mark.parametrize("with_finalizer", [False, True])
async def test_worker_nested_session_keeps_outer_settlement_owner(
    store, monkeypatch, with_finalizer
):
    from tests.core.test_verified_task_worker import _RecordingProvider

    from cayu import AgentSpec, Message, RunRequest
    from cayu.tasks.worker import run_task_worker

    app = CayuApp(task_store=store, enable_logging=False)
    provider = _RecordingProvider()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
    creation = request()
    if not with_finalizer:
        creation = creation.model_copy(update={"finalizer_task_id": None})
    await app.create_task_group(creation)
    owner_return = asyncio.Event()
    finalizer_done = asyncio.Event()
    original = type(store)._observe_task_group_invocation

    async def observe(instance, observation):
        if observation.owner_settled:
            owner_return.set()
            if with_finalizer:
                await asyncio.wait_for(finalizer_done.wait(), 10)
        await original(instance, observation)

    monkeypatch.setattr(type(store), "_observe_task_group_invocation", observe)
    after_session = []

    async def handler(_app, task, worker_id):
        async for _ in app.run(
            RunRequest(
                session_id="nested-worker-session",
                task_id=task.id,
                task_worker_id=worker_id,
                task_lease_expires_at=task.lease_expires_at,
                agent_name="worker",
                messages=[Message.text("user", "Work")],
            )
        ):
            pass
        snapshot = await app.load_task_group("race")
        execution = next(item for item in snapshot.quiescence.executions if item.task_id == task.id)
        assert execution.worker_id == worker_id
        assert execution.invocation is None and execution.settled_at is None
        after_session.append(task.id)

    async def finalize():
        await asyncio.wait_for(owner_return.wait(), 10)
        await store.complete_task("finalize", {})
        finalizer_done.set()

    finalizer = asyncio.create_task(finalize()) if with_finalizer else None
    try:
        assert (
            await asyncio.wait_for(
                run_task_worker(
                    app, store, handler, worker_id="owner", query=TaskQuery(type="b"), max_tasks=1
                ),
                20,
            )
            == 1
        )
        if finalizer is not None:
            await finalizer
        assert after_session == ["b"] and len(provider.requests) == 1
        snapshot = await app.load_task_group("race")
        execution = next(item for item in snapshot.quiescence.executions if item.task_id == "b")
        assert execution.invocation is None and execution.settled_at is not None
        assert (await store.load_task("b")).status is TaskStatus.COMPLETED
        assert (await store.load_task("a")).status is TaskStatus.CANCELLED
    finally:
        if finalizer is not None:
            finalizer.cancel()
            await asyncio.gather(finalizer, return_exceptions=True)


@pytest.mark.parametrize("invalid_claim", [False, True])
async def test_group_election_during_verifier_admission_keeps_worker_running(
    store, monkeypatch, invalid_claim
):
    from tests.core.test_verified_task_worker import (
        RecordingVerifier,
        _accepted_decision,
        _contract,
        _RecordingProvider,
        _Resolver,
        _StaticHandler,
        _task_result,
    )

    from cayu import AgentSpec
    from cayu.runtime.verified_task_worker import VerifiedTaskWorker
    from cayu.tasks.contracts import WorkCompletionConflict

    app = CayuApp(task_store=store, enable_logging=False)
    app.register_provider(_RecordingProvider(), default=True)
    app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
    contract = await store.publish_work_contract(_contract())
    verifier = RecordingVerifier(_accepted_decision())
    app.register_completion_verifier(contract.verifier, verifier)
    app.register_completion_result_resolver(contract.result_resolver, _Resolver(_task_result()))
    creation = request()
    nodes = tuple(
        node.model_copy(
            update={"task": node.task.model_copy(update={"work_contract": contract.reference()})}
        )
        if node.task.task_id == "b"
        else node
        for node in creation.graph.nodes
    )
    await app.create_task_group(
        creation.model_copy(update={"graph": creation.graph.model_copy(update={"nodes": nodes})})
    )
    entered, release = asyncio.Event(), asyncio.Event()
    claim = type(store).claim_completion_verification
    refused_proposals = []

    async def claim_after_election(instance, claim_request):
        proposal = await instance.load_completion_proposal(claim_request.proposal_id)
        if proposal.task_id == "b":
            refused_proposals.append(proposal.proposal_id)
            entered.set()
            await release.wait()
            if invalid_claim:
                claim_request = claim_request.model_copy(
                    update={"verifier_profile_fingerprint": "0" * 64}
                )
        return await claim(instance, claim_request)

    monkeypatch.setattr(type(store), "claim_completion_verification", claim_after_election)
    async with VerifiedTaskWorker(
        app, _StaticHandler(), worker_id="verified", query=TaskQuery(type="b"), poll_interval_s=0.01
    ) as owner:
        running = asyncio.create_task(owner.run(max_tasks=2))
        try:
            await asyncio.wait_for(entered.wait(), 15)
            await store.complete_task("a", {})
            # A completed invocation must never use the new no-entry proof,
            # even with otherwise exact authority and a real group election.
            from cayu.runtime.work_attempt_lifecycle import (
                WorkAttemptLifecycleSettlement,
                WorkAttemptPreEntrySettlementEvidence,
                pre_entry_settlement_authority,
            )
            from cayu.tasks.admission import WorkAttemptAdmissionConflict

            admitted = await store.load_latest_work_attempt_admission("b")
            assert admitted.execution_entry is not None
            with pytest.raises(WorkAttemptAdmissionConflict, match="undispatched owner"):
                await store.settle_work_attempt_lifecycle(
                    WorkAttemptLifecycleSettlement(
                        settlement_id="invalid-pre-entry-proof",
                        task_id="b",
                        admission_id=admitted.admission_id,
                        expected_admission_sha256=pre_entry_settlement_authority(admitted),
                        release_evidence=WorkAttemptPreEntrySettlementEvidence(
                            session_id=admitted.session_id,
                            session_instance_id=admitted.session_invocation.session_instance_id,
                            interaction_id=admitted.interaction_id,
                            profile_fingerprint=admitted.source_execution_profile_fingerprint,
                        ),
                        kind="group_cancellation",
                        stop_reason="work_contract_group_cancelled",
                    )
                )
            # Queue independent work behind the refused member on this same worker.
            await app.create_task(
                TaskCreate(task_id="independent", type="b", work_contract=contract.reference())
            )
            release.set()
            if invalid_claim:
                with pytest.raises(WorkCompletionConflict, match="exact prepared verifier profile"):
                    await asyncio.wait_for(running, 20)
                assert not verifier.requests
                assert (
                    await app.load_task_group("race")
                ).quiescence.status is TaskGroupQuiescenceStatus.DRAINING
                return
            assert await asyncio.wait_for(running, 20) == 2
            assert (await store.load_task("b")).status is TaskStatus.CANCELLED
            assert (await store.load_task("independent")).status is TaskStatus.COMPLETED
            assert len(verifier.requests) == 1
            assert verifier.requests[0].proposal.proposal_id not in refused_proposals
            snapshot = await app.load_task_group("race")
            assert snapshot.quiescence.status is TaskGroupQuiescenceStatus.QUIESCENT
            events = await app.list_task_group_events("race")
            assert sum(e.type is TaskGroupEventType.FINALIZER_RELEASED for e in events) == 1
            assert await store.claim_task("finalizer", TaskQuery(type="finalize")) is not None
        finally:
            release.set()
            if not running.done():
                running.cancel()
            await asyncio.gather(running, return_exceptions=True)


@pytest.mark.parametrize(
    "publication_fault",
    [
        "none",
        "precommit",
        "acknowledgement",
        "admission_read",
        "group_read",
        "receipt_read",
        "renewal",
        "cancel_read",
        "session_precommit",
        "session_acknowledgement",
        "release_precommit",
        "release_acknowledgement",
        "cancel_session",
        "marker_precommit",
        "marker_acknowledgement",
        "marker_conflict",
        "marker_malformed",
        "initial_precommit",
        "initial_acknowledgement",
        "sink_failure",
        "sink_precommit",
        "sink_acknowledgement",
    ],
)
async def test_group_election_before_execution_entry_settles_without_dispatch(
    store, monkeypatch, publication_fault, tmp_path
):
    from tests.core.test_verified_task_worker import (
        RecordingVerifier,
        _accepted_decision,
        _contract,
        _RecordingProvider,
        _Resolver,
        _StaticHandler,
        _task_result,
    )

    from cayu import AgentSpec
    from cayu.events import EventType
    from cayu.observability.events import EventSink
    from cayu.runtime.execution_profiles import (
        active_invocation_execution_profile_from_checkpoint,
        active_invocation_execution_profile_is_released,
    )
    from cayu.runtime.verified_task_worker import VerifiedTaskWorker
    from cayu.sessions import EventQuery, SessionStatus
    from cayu.storage.sqlite import SQLiteSessionStore
    from cayu.tasks.admission import WorkAttemptExecutionClaimLost

    sessions = SQLiteSessionStore(tmp_path / "pre-entry-sessions.sqlite")
    sink_calls = []

    class FailedSink(EventSink):
        async def emit(self, event):
            if event.type is EventType.INTERACTION_STARTED:
                sink_calls.append(event.id)
                if len(sink_calls) == 1:
                    raise RuntimeError("Ordinary returned sink failure")

    app = CayuApp(
        task_store=store,
        session_store=sessions,
        enable_logging=False,
        event_sinks=(FailedSink(),) if publication_fault.startswith("sink_") else (),
    )
    provider = _RecordingProvider()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
    contract = await store.publish_work_contract(_contract())
    verifier = RecordingVerifier(_accepted_decision())
    app.register_completion_verifier(contract.verifier, verifier)
    app.register_completion_result_resolver(contract.result_resolver, _Resolver(_task_result()))
    creation = request()
    nodes = tuple(
        node.model_copy(
            update={"task": node.task.model_copy(update={"work_contract": contract.reference()})}
        )
        if node.task.task_id == "b"
        else node
        for node in creation.graph.nodes
    )
    await app.create_task_group(
        creation.model_copy(update={"graph": creation.graph.model_copy(update={"nodes": nodes})})
    )
    entered, release = asyncio.Event(), asyncio.Event()
    enter = type(store).enter_work_attempt_execution
    settle = type(store).settle_work_attempt_lifecycle
    entries = []
    publications = []
    refused = False
    lookup_blocked, lookup_release = asyncio.Event(), asyncio.Event()
    injected = []
    session_writes = []
    retained_markers = []
    if publication_fault.startswith("marker_"):
        transform_checkpoint = SQLiteSessionStore.transform_checkpoint

        async def faulty_marker_clear(instance, session_id, transform):
            intercepted = False

            def wrapped(session, checkpoint):
                nonlocal intercepted
                updated = transform(session, checkpoint)
                if (
                    not session_writes
                    and session.status is SessionStatus.INTERRUPTED
                    and "pending_session_interrupt" in (checkpoint or {})
                    and "pending_session_interrupt" not in (updated or {})
                ):
                    intercepted = True
                    session_writes.append("marker")
                    retained_markers.append(checkpoint["pending_session_interrupt"])
                    if publication_fault == "marker_precommit":
                        raise RuntimeError("Injected marker clear failure")
                    if publication_fault in {"marker_conflict", "marker_malformed"}:
                        updated = dict(checkpoint)
                        updated["pending_session_interrupt"] = (
                            None
                            if publication_fault == "marker_malformed"
                            else {"interruption_request_id": "another-stop"}
                        )
                return updated

            result = await transform_checkpoint(instance, session_id, wrapped)
            if intercepted and publication_fault != "marker_precommit":
                raise RuntimeError("Injected marker clear acknowledgement loss")
            return result

        monkeypatch.setattr(SQLiteSessionStore, "transform_checkpoint", faulty_marker_clear)
    if (
        publication_fault.startswith(("session_", "release_", "initial_", "sink_"))
        or publication_fault == "cancel_session"
    ):
        session_method = (
            "retire_failed_first_event_delivery"
            if publication_fault.startswith("sink_")
            else "replace_initial_transcript_messages"
            if publication_fault.startswith("initial_")
            else "publish_interaction_transition"
            if publication_fault.startswith("session_")
            else "release_session_invocation"
        )
        session_write = getattr(SQLiteSessionStore, session_method)

        async def faulty_session_write(instance, *args, **kwargs):
            session_writes.append(session_method)
            fail = len(session_writes) == 1
            if fail and publication_fault == "cancel_session":
                lookup_blocked.set()
                await lookup_release.wait()
            if fail and publication_fault.endswith("precommit"):
                raise RuntimeError("Injected session cleanup failure")
            result = await session_write(instance, *args, **kwargs)
            if fail and publication_fault.endswith("acknowledgement"):
                raise RuntimeError("Injected session cleanup acknowledgement loss")
            return result

        monkeypatch.setattr(SQLiteSessionStore, session_method, faulty_session_write)

    async def gated_entry(instance, entry):
        nonlocal refused
        admission = await instance.load_work_attempt_admission(entry.admission_id)
        if admission.task_id == "b":
            entries.append(entry)
            entered.set()
            await release.wait()
        try:
            return await enter(instance, entry)
        finally:
            if admission.task_id == "b":
                refused = True

    async def faulty_settlement(instance, settlement):
        if settlement.task_id == "b":
            from cayu.tasks.admission import WorkAttemptAdmissionConflict

            # Exact admission content, not the absence of an entry alone,
            # authorizes cleanup. Conflicting proof must leave the fence intact.
            with pytest.raises(WorkAttemptAdmissionConflict):
                await settle(
                    instance, settlement.model_copy(update={"expected_admission_sha256": "0" * 64})
                )
            publications.append(settlement)
            if len(publications) == 1 and publication_fault == "precommit":
                raise RuntimeError("Injected pre-entry settlement failure")
        receipt = await settle(instance, settlement)
        if (
            settlement.task_id == "b"
            and len(publications) == 1
            and publication_fault == "acknowledgement"
        ):
            raise RuntimeError("Injected pre-entry acknowledgement loss")
        return receipt

    monkeypatch.setattr(type(store), "enter_work_attempt_execution", gated_entry)
    monkeypatch.setattr(type(store), "settle_work_attempt_lifecycle", faulty_settlement)
    owner = VerifiedTaskWorker(
        app, _StaticHandler(), worker_id="verified", query=TaskQuery(type="b"), poll_interval_s=0.01
    )
    method = {
        "admission_read": "load_work_attempt_admission",
        "group_read": "_task_group_cancellation_requested",
        "receipt_read": "load_work_attempt_lifecycle_receipt",
        "renewal": "renew_work_attempt_execution_claim",
        "cancel_read": "_task_group_cancellation_requested",
    }.get(publication_fault)
    if method is not None:
        original = getattr(type(store), method)

        async def fail_before_publication(instance, *args, **kwargs):
            if refused and not injected:
                injected.append(method)
                if publication_fault == "cancel_read":
                    lookup_blocked.set()
                    await lookup_release.wait()
                else:
                    raise RuntimeError("Injected failure before pre-entry publication")
            return await original(instance, *args, **kwargs)

        monkeypatch.setattr(type(store), method, fail_before_publication)
    running = asyncio.create_task(owner.run(max_tasks=1))
    try:
        await asyncio.wait_for(entered.wait(), 15)
        await store.complete_task("a", {})
        release.set()
        if publication_fault in {"cancel_read", "cancel_session"}:
            await asyncio.wait_for(lookup_blocked.wait(), 15)
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await running
            assert running.cancelled() and running.cancelling() == 1
            assert not owner._pre_entry_publication.done()
            assert (
                await app.load_task_group("race")
            ).quiescence.status is TaskGroupQuiescenceStatus.DRAINING
            lookup_release.set()
            await owner.aclose()
            owner = VerifiedTaskWorker(
                app,
                _StaticHandler(),
                worker_id="verified",
                query=TaskQuery(type="b"),
                poll_interval_s=0.01,
            )
        elif publication_fault.startswith(("session_", "release_", "marker_", "initial_", "sink_")):
            try:
                assert await asyncio.wait_for(running, 20) == 1
            except BaseExceptionGroup:
                assert not publications
                assert (
                    await app.load_task_group("race")
                ).quiescence.status is TaskGroupQuiescenceStatus.DRAINING
                if publication_fault in {"marker_conflict", "marker_malformed"}:
                    from cayu.sessions import SessionRunFenced

                    admission = await store.load_work_attempt_admission(entries[0].admission_id)
                    before_retry = await sessions.load_checkpoint(admission.session_id)
                    with pytest.raises(SessionRunFenced, match="marker belongs"):
                        await owner.run(max_tasks=0)
                    assert await sessions.load_checkpoint(admission.session_id) == before_retry
                    assert (
                        await store.load_work_attempt_lifecycle_receipt(admission.admission_id)
                        is None
                    )
                    await sessions.transform_checkpoint(
                        admission.session_id,
                        lambda _session, checkpoint: {
                            **checkpoint,
                            "pending_session_interrupt": retained_markers[0],
                        },
                    )
                assert await owner.run(max_tasks=0) == 0
            assert session_writes
        elif publication_fault in {"precommit", "admission_read", "group_read", "receipt_read"}:
            with pytest.raises(BaseExceptionGroup):
                await asyncio.wait_for(running, 20)
            assert (
                await app.load_task_group("race")
            ).quiescence.status is TaskGroupQuiescenceStatus.DRAINING
            # Retry only the retained acknowledgement, not preparation or execution.
            assert await owner.run(max_tasks=0) == 0
            if publication_fault == "precommit":
                assert publications[0] == publications[1]
            else:
                assert len(publications) == 1 and len(injected) == 1
        else:
            assert await asyncio.wait_for(running, 20) == 1
        if publication_fault == "renewal":
            # Cleanup-only settlement must not require another live lease.
            assert not injected
            monkeypatch.setattr(type(store), method, original)
        assert not verifier.requests
        assert not provider.requests
        assert len(entries) == 1
        admission = await store.load_work_attempt_admission(entries[0].admission_id)
        assert admission.execution_entry is None
        session = await sessions.load(admission.session_id)
        assert session.status is SessionStatus.INTERRUPTED
        checkpoint = await sessions.load_checkpoint(session.id)
        assert "pending_session_interrupt" not in (checkpoint or {})
        active = active_invocation_execution_profile_from_checkpoint(checkpoint)
        assert active_invocation_execution_profile_is_released(
            active, session_id=session.id, run_epoch=session.run_epoch
        )
        assert await sessions.load_deferred_interaction_input(session.id) is None
        records = await sessions.query_events(EventQuery(session_id=session.id))
        assert sum(r.event.type is EventType.INTERACTION_INTERRUPTED for r in records) == 1
        assert sum(r.event.type is EventType.SESSION_INTERRUPTED for r in records) == 1
        if publication_fault.startswith("sink_"):
            assert len(sink_calls) == 1
            started = next(
                r.event for r in records if r.event.type is EventType.INTERACTION_STARTED
            )
            delivery = await sessions.get_persisted_event_side_effect_delivery(
                session_id=session.id, event_id=started.id
            )
            assert delivery.status.value == "dead_lettered" and delivery.attempts == 1
            assert "Ordinary returned sink failure" in delivery.last_error
            assert (
                await sessions.claim_persisted_event_side_effect(
                    session_id=session.id, event_id=started.id
                )
                is None
            )
        from cayu.sessions import RunnerObservedEventIdentity

        await sessions.load_runner_owned_interrupted_evidence(
            session.id,
            observed_events=tuple(
                RunnerObservedEventIdentity(
                    session_id=session.id, sequence=r.sequence, event_type=r.event.type
                )
                for r in records
            ),
        )
        reopened = SQLiteSessionStore(tmp_path / "pre-entry-sessions.sqlite")
        try:
            assert await reopened.load(session.id) == session
            assert await reopened.query_events(EventQuery(session_id=session.id)) == records
        finally:
            await reopened.close()
        assert (await store.load_task("b")).status is TaskStatus.CANCELLED
        with pytest.raises(WorkAttemptExecutionClaimLost):
            await enter(store, entries[0])
        receipt = await store.load_work_attempt_lifecycle_receipt(admission.admission_id)
        assert await settle(store, receipt.request) == receipt
        events = await app.list_task_group_events("race")
        assert sum(e.type is TaskGroupEventType.FINALIZER_RELEASED for e in events) == 1
        await app.create_task(
            TaskCreate(task_id="independent", type="b", work_contract=contract.reference())
        )
        assert await owner.run(max_tasks=1) == 1
        assert (await store.load_task("independent")).status is TaskStatus.COMPLETED
    finally:
        lookup_release.set()
        release.set()
        if not running.done():
            running.cancel()
        await asyncio.gather(running, return_exceptions=True)
        await owner.aclose()
        await sessions.close()


@pytest.mark.parametrize("takeover", [False, True])
async def test_pre_entry_recovery_waits_for_admission_sink_return(
    store, tmp_path, monkeypatch, takeover
):
    from datetime import UTC, datetime

    from tests.core.test_verified_task_worker import _contract, _RecordingProvider, _StaticHandler

    from cayu import AgentSpec, Message, RunRequest
    from cayu.events import EventType
    from cayu.runtime.verified_task_worker import VerifiedTaskWorker
    from cayu.sessions import EventQuery, SessionStatus
    from cayu.storage.sqlite import SQLiteSessionStore
    from cayu.tasks.admission import WorkAttemptExecutionClaimLost, WorkAttemptExecutionRequest

    entered, renewed, checked = asyncio.Event(), asyncio.Event(), asyncio.Event()
    release = threading.Event()
    effects = []

    from cayu.observability.events import EventSink

    class Sink(EventSink):
        async def emit(self, event):
            if event.type is EventType.INTERACTION_STARTED:
                entered.set()
                await asyncio.to_thread(release.wait)
                effects.append("returned")

    sessions = SQLiteSessionStore(tmp_path / "handoff.sqlite")
    claim_delivery = sessions.claim_persisted_event_side_effect

    async def short_delivery_claim(**kwargs):
        return await claim_delivery(**{**kwargs, "lease_seconds": 0.1})

    monkeypatch.setattr(sessions, "claim_persisted_event_side_effect", short_delivery_claim)
    app = CayuApp(
        task_store=store, session_store=sessions, event_sinks=(Sink(),), enable_logging=False
    )
    provider = _RecordingProvider()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
    contract = await store.publish_work_contract(_contract())
    creation = request()
    nodes = tuple(
        node.model_copy(
            update={"task": node.task.model_copy(update={"work_contract": contract.reference()})}
        )
        if node.task.task_id == "b"
        else node
        for node in creation.graph.nodes
    )
    await app.create_task_group(
        creation.model_copy(update={"graph": creation.graph.model_copy(update={"nodes": nodes})})
    )

    async def fail_renewal(instance, request):
        renewed.set()
        raise WorkAttemptExecutionClaimLost("Injected handoff renewal failure")

    monkeypatch.setattr(type(store), "renew_work_attempt_execution_claim", fail_renewal)
    original = asyncio.create_task(
        app.admit_work_attempt(
            RunRequest(
                agent_name="worker",
                task_id="b",
                session_id="handoff",
                messages=[Message.text("user", "Never dispatch losing work.")],
            ),
            execution=WorkAttemptExecutionRequest(
                admission_id="handoff",
                claim_id="handoff",
                attempt_id="handoff",
                interaction_id="handoff",
                worker_id="original",
                generation=1,
                lease_seconds=5,
            ),
        )
    )
    peer = CayuApp(task_store=store, session_store=sessions, enable_logging=False)
    peer.register_provider(provider, default=True)
    peer.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
    worker = VerifiedTaskWorker(
        peer,
        _StaticHandler(),
        worker_id="recovery",
        query=TaskQuery(type="b"),
        poll_interval_s=0.01,
    )
    proof = peer._session_engine.settle_work_attempt_admission_handoff
    stop = asyncio.Event()

    async def observe(admission):
        result = await proof(admission)
        if not release.is_set() or takeover:
            assert not result
            checked.set()
            stop.set()
        return result

    monkeypatch.setattr(peer._session_engine, "settle_work_attempt_admission_handoff", observe)
    try:
        async with asyncio.timeout(20):
            await entered.wait()
            await renewed.wait()
            await store.complete_task("a", {})
            admission = await store.load_work_attempt_admission("handoff")
            while datetime.now(UTC) <= admission.claim.lease_expires_at:
                await asyncio.sleep(0.01)
            if takeover:
                records = await sessions.query_events(
                    EventQuery(session_id="handoff", event_type=EventType.INTERACTION_STARTED)
                )
                await peer._session_engine._event_writer.fan_out_persisted([records[0].event])
                delivery = await sessions.get_persisted_event_side_effect_delivery(
                    session_id="handoff", event_id=records[0].event.id
                )
                assert delivery.status.value == "delivered" and delivery.attempts == 2
            assert not original.done()
            assert await worker.run(stop=stop) == 0
            assert checked.is_set() and not effects
            assert (await sessions.load("handoff")).status is SessionStatus.RUNNING
            assert await store.load_work_attempt_lifecycle_receipt("handoff") is None
            assert await store.claim_task("finalizer", TaskQuery(type="finalize")) is None
            assert (
                await app.load_task_group("race")
            ).quiescence.status is TaskGroupQuiescenceStatus.DRAINING
            release.set()
            if takeover:
                outcome = await asyncio.gather(original, return_exceptions=True)
                assert isinstance(outcome[0], Exception)
                stop.clear()
                assert await worker.run(stop=stop) == 0
                assert await store.load_work_attempt_lifecycle_receipt("handoff") is None
                assert await store.claim_task("finalizer", TaskQuery(type="finalize")) is None
                assert not provider.requests and effects == ["returned"]
                return
            with pytest.raises(WorkAttemptExecutionClaimLost):
                await original
            assert effects == ["returned"]
            assert await worker.run(max_tasks=1) == 1
        assert not provider.requests
        assert (await store.load_task("b")).status is TaskStatus.CANCELLED
        assert (await sessions.load("handoff")).status is SessionStatus.INTERRUPTED
        events = await app.list_task_group_events("race")
        assert sum(event.type is TaskGroupEventType.FINALIZER_RELEASED for event in events) == 1
    finally:
        release.set()
        await asyncio.gather(original, return_exceptions=True)
        await worker.aclose()
        await sessions.close()


@pytest.mark.parametrize("terminal_type", ["interaction.interrupted", "session.interrupted"])
@pytest.mark.parametrize("sink_outcome", ["returned", "failed", "cancelled_owner"])
async def test_pre_entry_cleanup_waits_for_competing_terminal_sink(
    store, tmp_path, terminal_type, sink_outcome
):
    from datetime import UTC, datetime

    from tests.core.test_verified_task_worker import _contract, _RecordingProvider, _StaticHandler

    from cayu import AgentSpec, Message, RunRequest
    from cayu.events import EventType
    from cayu.observability.events import EventSink
    from cayu.runtime.execution_profiles import (
        active_invocation_execution_profile_from_checkpoint,
        active_invocation_execution_profile_is_released,
    )
    from cayu.runtime.verified_task_worker import VerifiedTaskWorker
    from cayu.sessions import SessionStatus
    from cayu.storage.sqlite import SQLiteSessionStore
    from cayu.tasks.admission import WorkAttemptExecutionRequest, WorkAttemptRecoveryRequired

    entered = asyncio.Event()
    release = threading.Event()
    effects = []

    class Sink(EventSink):
        async def emit(self, event):
            if event.type == terminal_type:
                entered.set()
                await asyncio.to_thread(release.wait)
                effects.append("returned")
                if sink_outcome == "failed":
                    raise RuntimeError("Returned terminal sink failure")

    path = tmp_path / "terminal-handoff.sqlite"
    sessions, peer_sessions = SQLiteSessionStore(path), SQLiteSessionStore(path)
    provider = _RecordingProvider()

    def application(session_store, sinks=()):
        app = CayuApp(
            task_store=store, session_store=session_store, event_sinks=sinks, enable_logging=False
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
        return app

    app, peer = application(sessions, (Sink(),)), application(peer_sessions)
    contract = await store.publish_work_contract(_contract())
    creation = request()
    nodes = tuple(
        node.model_copy(
            update={"task": node.task.model_copy(update={"work_contract": contract.reference()})}
        )
        if node.task.task_id == "b"
        else node
        for node in creation.graph.nodes
    )
    await app.create_task_group(
        creation.model_copy(update={"graph": creation.graph.model_copy(update={"nodes": nodes})})
    )
    admission = await app.admit_work_attempt(
        RunRequest(
            agent_name="worker",
            task_id="b",
            session_id="terminal-handoff",
            messages=[Message.text("user", "Do not dispatch losing work.")],
        ),
        execution=WorkAttemptExecutionRequest(
            admission_id="terminal-handoff",
            claim_id="terminal-handoff",
            attempt_id="terminal-handoff",
            interaction_id="terminal-handoff",
            worker_id="original",
            generation=1,
            lease_seconds=5,
        ),
    )
    await store.complete_task("a", {})
    owner = VerifiedTaskWorker(
        app, _StaticHandler(), worker_id="cleanup", query=TaskQuery(type="b"), poll_interval_s=0.01
    )
    recovery = VerifiedTaskWorker(
        peer, _StaticHandler(), worker_id="peer", query=TaskQuery(type="b"), poll_interval_s=0.01
    )
    running = None
    try:
        async with asyncio.timeout(25):
            while datetime.now(UTC) <= admission.claim.lease_expires_at:
                await asyncio.sleep(0.01)
            running = asyncio.create_task(owner.run(max_tasks=1))
            await entered.wait()
            if sink_outcome == "cancelled_owner":
                running.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await running
                assert running.cancelled() and running.cancelling() == 1
                assert owner._pre_entry_publication is not None
                assert not owner._pre_entry_publication.done()
            with pytest.raises(
                WorkAttemptRecoveryRequired, match="terminal handoff is not settled"
            ):
                await recovery.run(max_tasks=1)
            assert (not running.done() or sink_outcome == "cancelled_owner") and not effects
            assert await store.load_work_attempt_lifecycle_receipt(admission.admission_id) is None
            assert await store.claim_task("finalizer", TaskQuery(type="finalize")) is None
            session = await peer_sessions.load(admission.session_id)
            assert session.status is SessionStatus.INTERRUPTED
            active = active_invocation_execution_profile_from_checkpoint(
                await peer_sessions.load_checkpoint(session.id)
            )
            assert not active_invocation_execution_profile_is_released(
                active, session_id=session.id, run_epoch=session.run_epoch
            )
            release.set()
            if sink_outcome == "cancelled_owner":
                await owner.aclose()
            else:
                assert await running == 1
            await recovery.aclose()
        assert effects == ["returned"] and not provider.requests
        assert (await store.load_task("b")).status is TaskStatus.CANCELLED
        events = await app.list_task_group_events("race")
        assert sum(event.type is TaskGroupEventType.FINALIZER_RELEASED for event in events) == 1
        assert await store.load_work_attempt_lifecycle_receipt(admission.admission_id) is not None
        from cayu.sessions import EventQuery

        for event_type in (EventType.INTERACTION_INTERRUPTED, EventType.SESSION_INTERRUPTED):
            records = await peer_sessions.query_events(
                EventQuery(session_id=admission.session_id, event_type=event_type)
            )
            assert len(records) == 1
            delivery = await peer_sessions.get_persisted_event_side_effect_delivery(
                session_id=admission.session_id, event_id=records[0].event.id
            )
            assert delivery.attempts == 1
            assert delivery.status.value == (
                "dead_lettered"
                if sink_outcome == "failed" and event_type == terminal_type
                else "delivered"
            )
            assert (
                await peer_sessions.claim_persisted_event_side_effect(
                    session_id=admission.session_id, event_id=records[0].event.id
                )
                is None
            )
    finally:
        release.set()
        if running is not None:
            await asyncio.gather(running, return_exceptions=True)
        await owner.aclose()
        await recovery.aclose()
        await sessions.close()
        await peer_sessions.close()


@pytest.mark.parametrize(
    "marker_failure",
    [None, "precommit", "acknowledgement", "before_delivery", "pending_claim_race"],
)
async def test_unentered_loser_session_cleanup_is_discoverable_after_restart(
    store, tmp_path, monkeypatch, marker_failure
):
    from datetime import UTC, datetime

    from tests.core.test_verified_task_worker import _contract, _RecordingProvider, _StaticHandler

    from cayu import AgentSpec, Message, RunRequest
    from cayu.events import EventType
    from cayu.runtime.execution_profiles import (
        active_invocation_execution_profile_from_checkpoint,
        active_invocation_execution_profile_is_released,
    )
    from cayu.runtime.verified_task_worker import VerifiedTaskWorker
    from cayu.sessions import EventQuery, SessionStatus
    from cayu.storage.sqlite import SQLiteSessionStore
    from cayu.tasks.admission import WorkAttemptExecutionRequest

    path = tmp_path / "restarted-pre-entry.sqlite"
    sessions = SQLiteSessionStore(path)

    def application(session_store):
        app = CayuApp(task_store=store, session_store=session_store, enable_logging=False)
        provider = _RecordingProvider()
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
        return app, provider

    app, original_provider = application(sessions)
    contract = await store.publish_work_contract(_contract())
    creation = request()
    nodes = tuple(
        node.model_copy(
            update={"task": node.task.model_copy(update={"work_contract": contract.reference()})}
        )
        if node.task.task_id == "b"
        else node
        for node in creation.graph.nodes
    )
    await app.create_task_group(
        creation.model_copy(update={"graph": creation.graph.model_copy(update={"nodes": nodes})})
    )
    admission = await app.admit_work_attempt(
        RunRequest(
            agent_name="worker",
            task_id="b",
            session_id="unentered",
            messages=[Message.text("user", "Never dispatch this losing work.")],
        ),
        execution=WorkAttemptExecutionRequest(
            admission_id="unentered-admission",
            claim_id="unentered-claim",
            attempt_id="unentered-attempt",
            interaction_id="unentered-interaction",
            worker_id="lost-worker",
            generation=1,
            lease_seconds=5,
        ),
    )
    await store.complete_task("a", {})
    await sessions.close()
    sessions = SQLiteSessionStore(path)
    app, provider = application(sessions)
    owner = VerifiedTaskWorker(
        app,
        _StaticHandler(),
        worker_id="replacement",
        query=TaskQuery(type="b"),
        poll_interval_s=0.01,
    )
    prior_owner = None
    prior_sessions = None
    try:
        async with asyncio.timeout(15):
            while datetime.now(UTC) <= admission.claim.lease_expires_at:
                await asyncio.sleep(0.05)
            if marker_failure is not None:
                clear = app._session_engine._clear_pending_session_interrupt

                async def fail_clear(*args, **kwargs):
                    if marker_failure == "acknowledgement":
                        await clear(*args, **kwargs)
                    raise RuntimeError("Injected restart marker publication failure")

                if marker_failure in {"before_delivery", "pending_claim_race"}:
                    fan_out = app._session_engine._event_writer.fan_out_persisted

                    async def fail_delivery(events):
                        if any(event.type is EventType.INTERACTION_INTERRUPTED for event in events):
                            raise RuntimeError("Injected restart before terminal fan-out")
                        return await fan_out(events)

                    monkeypatch.setattr(
                        app._session_engine._event_writer, "fan_out_persisted", fail_delivery
                    )
                else:
                    monkeypatch.setattr(
                        app._session_engine, "_clear_pending_session_interrupt", fail_clear
                    )
                with pytest.raises(RuntimeError, match="Injected restart"):
                    await owner.run(max_tasks=1)
                assert (
                    await app.load_task_group("race")
                ).quiescence.status is TaskGroupQuiescenceStatus.DRAINING
                assert (
                    await store.load_work_attempt_lifecycle_receipt(admission.admission_id) is None
                )
                prior_owner, prior_sessions = owner, sessions
                await sessions.close()
                sessions = SQLiteSessionStore(path)
                app, provider = application(sessions)
                owner = VerifiedTaskWorker(
                    app,
                    _StaticHandler(),
                    worker_id="restarted",
                    query=TaskQuery(type="b"),
                    poll_interval_s=0.01,
                )
            if marker_failure == "pending_claim_race":
                from cayu.observability.events import EventSink
                from cayu.runtime import _event_writer

                monkeypatch.setattr(_event_writer, "_PERSISTED_SIDE_EFFECT_RETRY_DELAY_SECONDS", 0)
                calls = []
                reached, release = asyncio.Event(), asyncio.Event()
                first_claim = sessions.claim_first_persisted_event_side_effect

                async def delayed_first_claim(expected):
                    if not reached.is_set():
                        reached.set()
                        await release.wait()
                    return await first_claim(expected)

                class ReturnedFailure(EventSink):
                    async def emit(self, event):
                        calls.append(event.type)
                        raise RuntimeError("First callback returned before recovery claimed")

                monkeypatch.setattr(
                    sessions, "claim_first_persisted_event_side_effect", delayed_first_claim
                )
                publisher = CayuApp(
                    session_store=sessions, event_sinks=(ReturnedFailure(),), enable_logging=False
                )
                running = asyncio.create_task(owner.run(max_tasks=1))
                try:
                    await reached.wait()
                    records = await sessions.query_events(
                        EventQuery(
                            session_id=admission.session_id,
                            event_type=EventType.INTERACTION_INTERRUPTED,
                        )
                    )
                    await publisher._session_engine._event_writer.fan_out_persisted(
                        [records[0].event]
                    )
                    release.set()
                    assert await running == 1
                    delivery = await sessions.get_persisted_event_side_effect_delivery(
                        session_id=admission.session_id, event_id=records[0].event.id
                    )
                    assert calls == [EventType.INTERACTION_INTERRUPTED]
                    assert delivery.attempts == 1 and delivery.status.value == "dead_lettered"
                finally:
                    release.set()
                    await asyncio.gather(running, return_exceptions=True)
            else:
                assert await owner.run(max_tasks=1) == 1
        assert not provider.requests and not original_provider.requests
        assert (await store.load_task("b")).status is TaskStatus.CANCELLED
        session = await sessions.load(admission.session_id)
        assert session.status is SessionStatus.INTERRUPTED
        assert "pending_session_interrupt" not in (await sessions.load_checkpoint(session.id) or {})
        active = active_invocation_execution_profile_from_checkpoint(
            await sessions.load_checkpoint(session.id)
        )
        assert active_invocation_execution_profile_is_released(
            active, session_id=session.id, run_epoch=session.run_epoch
        )
        events = await sessions.query_events(EventQuery(session_id=session.id))
        from cayu.sessions import RunnerObservedEventIdentity

        await sessions.load_runner_owned_interrupted_evidence(
            session.id,
            observed_events=tuple(
                RunnerObservedEventIdentity(
                    session_id=session.id, sequence=r.sequence, event_type=r.event.type
                )
                for r in events
            ),
        )
        assert sum(r.event.type is EventType.INTERACTION_INTERRUPTED for r in events) == 1
        assert sum(r.event.type is EventType.SESSION_INTERRUPTED for r in events) == 1
        group_events = await app.list_task_group_events("race")
        assert sum(e.type is TaskGroupEventType.FINALIZER_RELEASED for e in group_events) == 1
        assert await owner.run(max_tasks=0) == 0
        assert await app.list_task_group_events("race") == group_events
    finally:
        await owner.aclose()
        if prior_owner is not None:
            await prior_owner.aclose()
            await prior_sessions.close()
        await sessions.close()


async def test_active_loser_retains_execution_obligation(store):
    app = CayuApp(task_store=store, enable_logging=False)
    await app.create_task_group(request())
    claimed = await store.claim_task("worker", TaskQuery(type="b"))
    await store.mark_claimed_task_execution_started(
        "b",
        "worker",
        claimed.lease_expires_at,
    )
    await store.complete_task("a", {})
    snapshot = await app.load_task_group("race")
    assert snapshot.quiescence.status is TaskGroupQuiescenceStatus.DRAINING
    assert snapshot.quiescence.unsettled_task_ids == ("b",)
    assert len(snapshot.quiescence.executions) == 1
    assert (await store.load_task("b")).status_reason == "cancellation_requested"
    assert await store.claim_task("finalizer-worker", TaskQuery(type="finalize")) is None
    with pytest.raises(ValueError):
        await store.complete_task(
            "b", {}, worker_id="worker", lease_expires_at=claimed.lease_expires_at
        )


async def test_changed_barrier_authority_conflicts_on_public_replay(store):
    app = CayuApp(task_store=store, enable_logging=False)
    await app.create_task_group(request())
    with pytest.raises(TaskGroupConflict):
        await app.create_task_group(request(timeout=30))


async def test_expiry_preserves_fence_until_exact_cancellation_reconciliation(store):
    from tests.core.task_terminalization_conformance import (
        ordinary_cancellation_reconciliation_request,
    )

    app = CayuApp(task_store=store, enable_logging=False)
    creation = request(timeout=10)
    nodes = tuple(
        node.model_copy(
            update={
                "task": node.task.model_copy(
                    update={
                        "metadata": {
                            "execution_profile_fingerprint": "b" * 64,
                            "effect_fingerprint": "c" * 64,
                        }
                    }
                )
            }
        )
        if node.task.task_id == "b"
        else node
        for node in creation.graph.nodes
    )
    creation = creation.model_copy(
        update={"graph": creation.graph.model_copy(update={"nodes": nodes})}
    )
    await app.create_task_group(creation)
    claim = await store.claim_task("lost-worker", TaskQuery(type="b"), lease_seconds=1)
    await store.mark_claimed_task_execution_started("b", "lost-worker", claim.lease_expires_at)
    await store.complete_task("a", {})
    await asyncio.sleep(1.05)
    await store.reclaim_expired()
    assert await store.claim_task("replacement", TaskQuery(type="b")) is None
    assert await store.claim_task("finalizer", TaskQuery(type="finalize")) is None
    before = await app.reconcile_task_group("race")
    assert before.quiescence.unsettled_task_ids == ("b",)
    cancellation = ordinary_cancellation_reconciliation_request(await store.load_task("b"))
    first = await store.reconcile_task_cancellation(cancellation)
    assert (
        await app.load_task_group("race")
    ).quiescence.status is TaskGroupQuiescenceStatus.QUIESCENT
    history = await app.list_task_group_events("race")
    assert await store.reconcile_task_cancellation(cancellation) == first
    assert await app.list_task_group_events("race") == history
    assert await store.claim_task("finalizer", TaskQuery(type="finalize")) is not None


@pytest.mark.parametrize("descendant", [False, True])
async def test_retry_reconciliation_settles_exact_group_execution(store, descendant):
    from tests.core.test_task_retry_series import _retry_cancellation_reconciliation_request

    app = CayuApp(task_store=store, enable_logging=False)
    await retry_group(app)
    claim = await store.claim_task("lost-worker", TaskQuery(type="b"), lease_seconds=1)
    if descendant:
        successor = (await schedule_retry(store, claim, "first-attempt")).successor
        assert successor is not None
        claim = await store.claim_task("lost-worker", TaskQuery(type="b"), lease_seconds=1)
        assert claim.id == successor.id
    await store.mark_claimed_task_execution_started(
        claim.id, claim.worker_id, claim.lease_expires_at
    )
    await store.complete_task("a", {})
    await asyncio.sleep(1.05)
    await store.reclaim_expired()
    assert await store.claim_task("finalizer", TaskQuery(type="finalize")) is None
    cancellation = _retry_cancellation_reconciliation_request(await store.load_task(claim.id))
    first = await store.reconcile_task_retry_cancellation(cancellation)
    snapshot = await app.load_task_group("race")
    assert snapshot.quiescence.status is TaskGroupQuiescenceStatus.QUIESCENT
    execution = next(item for item in snapshot.quiescence.executions if item.task_id == claim.id)
    assert execution.worker_id == claim.worker_id and execution.settled_at is not None
    history = await app.list_task_group_events("race")
    reopened = (
        SQLiteTaskStore(store.path)
        if isinstance(store, SQLiteTaskStore)
        else PostgresTaskStore(_postgres_address(store))
        if isinstance(store, PostgresTaskStore)
        else store
    )
    try:
        assert await reopened.reconcile_task_retry_cancellation(cancellation) == first
        assert await reopened.load_task_group("race") == snapshot
        assert await reopened.list_task_group_events("race") == history
        assert await reopened.claim_task("finalizer", TaskQuery(type="finalize")) is not None
    finally:
        if reopened is not store:
            await reopened.close()


async def test_idle_worker_discovers_timeout_without_dispatch(store):
    from cayu.tasks.worker import run_task_worker

    app = CayuApp(task_store=store, enable_logging=False)
    await app.create_task_group(request(timeout=0.02))
    claimed = await store.claim_task("lost-worker", TaskQuery(type="b"))
    await store.mark_claimed_task_execution_started("b", "lost-worker", claimed.lease_expires_at)
    await store.complete_task("a", {})
    assert await store.list_task_group_reconciliation_candidates(limit=1) == ["race"]
    assert await store.list_task_group_reconciliation_candidates(after_group_id="race") == []
    await asyncio.sleep(0.03)
    # Inspection alone never performs the timeout transition.
    assert (
        await app.load_task_group("race")
    ).quiescence.status is TaskGroupQuiescenceStatus.DRAINING
    stop = asyncio.Event()

    async def handler(*args):
        pytest.fail("Maintenance must not dispatch fenced group work.")

    worker = asyncio.create_task(
        run_task_worker(
            app,
            store,
            handler,
            worker_id="maintenance",
            query=TaskQuery(type="finalize"),
            poll_interval_s=0.01,
            stop=stop,
        )
    )
    try:
        async with asyncio.timeout(5):
            while (
                await app.load_task_group("race")
            ).quiescence.status is TaskGroupQuiescenceStatus.DRAINING:
                await asyncio.sleep(0.01)
        assert await store.list_task_group_reconciliation_candidates() == []
        assert await store.claim_task("finalizer", TaskQuery(type="finalize")) is None
        events = await app.list_task_group_events("race")
        assert sum(e.type is TaskGroupEventType.TIMEOUT for e in events) == 1
        await app.reconcile_task_group("race")
        assert await app.list_task_group_events("race") == events
    finally:
        stop.set()
        assert await asyncio.wait_for(worker, 5) == 0


async def test_idle_dispatcher_observes_group_timeout_without_extra_runtime_ports(store):
    from tests.core.test_dispatch import _SecretFreeDispatchRuntime

    from cayu.tasks.dispatch import TaskStoreDispatcher

    app = CayuApp(task_store=store, enable_logging=False)
    await app.create_task_group(request(timeout=0.02))
    claim = await store.claim_task("lost", TaskQuery(type="b"))
    await store.mark_claimed_task_execution_started("b", "lost", claim.lease_expires_at)
    await store.complete_task("a", {})
    await asyncio.sleep(0.03)
    dispatcher = TaskStoreDispatcher(store)
    assert await dispatcher.process_next(_SecretFreeDispatchRuntime(), worker_id="observer") is None
    assert (
        await app.load_task_group("race")
    ).quiescence.status is TaskGroupQuiescenceStatus.ATTENTION_REQUIRED
    assert await store.claim_task("finalizer", TaskQuery(type="finalize")) is None
    events = await app.list_task_group_events("race")
    assert sum(event.type is TaskGroupEventType.TIMEOUT for event in events) == 1
    assert await dispatcher.process_next(_SecretFreeDispatchRuntime(), worker_id="observer") is None
    assert await app.list_task_group_events("race") == events


@pytest.mark.parametrize(
    "missing",
    [
        "_settle_task_group_execution",
        "_observe_task_group_invocation",
        "_observe_task_group_result_resolution",
        "mark_claimed_task_execution_started",
        "_task_group_cancellation_requested",
        "_task_group_retains_execution",
        "list_task_group_reconciliation_candidates",
        "resolve_task_group_quiescence",
    ],
)
async def test_partial_store_group_capability_rejected_before_admission(missing):
    from cayu.tasks.base import TaskStore

    partial_store = type(
        "PartialGroupStore", (InMemoryTaskStore,), {missing: getattr(TaskStore, missing)}
    )()
    app = CayuApp(task_store=partial_store, enable_logging=False)
    with pytest.raises(NotImplementedError, match="capable store"):
        await app.create_task_group(request())
    assert await partial_store.load_task("a") is None
    assert await partial_store.load_task_group("race") is None


async def test_expired_undispatched_claim_can_settle_without_effect_reconciliation(store):
    app = CayuApp(task_store=store, enable_logging=False)
    await app.create_task_group(request(timeout=10))
    await store.claim_task("lost-worker", TaskQuery(type="b"), lease_seconds=1)
    await store.complete_task("a", {})
    assert (
        await app.load_task_group("race")
    ).quiescence.status is TaskGroupQuiescenceStatus.DRAINING
    assert await store.claim_task("finalizer", TaskQuery(type="finalize")) is None
    await asyncio.sleep(1.05)
    assert (
        await app.reconcile_task_group("race")
    ).quiescence.status is TaskGroupQuiescenceStatus.QUIESCENT
    assert (await store.load_task("b")).status is TaskStatus.CANCELLED
    assert await store.claim_task("finalizer", TaskQuery(type="finalize")) is not None


async def test_local_effect_record_survives_task_cancellation_and_blocks_finalizer(store):
    from tests.core.task_terminalization_conformance import (
        ordinary_cancellation_reconciliation_request,
    )
    from tests.core.test_local_execution_attempts import _receipt, _request, _settlement, _start

    from cayu.runtime.local_execution_attempts import (
        LocalExecutionAttemptConflict,
        build_local_execution_attempt_authority,
    )

    app = CayuApp(task_store=store, enable_logging=False)
    creation = request(timeout=10)
    nodes = tuple(
        node.model_copy(
            update={
                "task": node.task.model_copy(
                    update={
                        "metadata": {
                            "execution_profile_fingerprint": "b" * 64,
                            "effect_fingerprint": "c" * 64,
                        }
                    }
                )
            }
        )
        if node.task.task_id == "b"
        else node
        for node in creation.graph.nodes
    )
    await app.create_task_group(
        creation.model_copy(
            update={
                "graph": creation.graph.model_copy(update={"nodes": nodes}),
            }
        )
    )
    claimed = await store.claim_task("worker-a", TaskQuery(type="b"), lease_seconds=1)
    authority = build_local_execution_attempt_authority(
        app=app,
        task=claimed,
        worker_id="worker-a",
        request=_request(),
    )
    await store.prepare_local_execution_attempt(authority)
    start = _start(authority, root=True)
    await store.start_local_execution_attempt(start)
    await store.complete_task("a", {})
    another_authority = build_local_execution_attempt_authority(
        app=app,
        task=claimed,
        worker_id="worker-a",
        request=_request().model_copy(update={"effect_lineage_id": "new-effect"}),
    )
    with pytest.raises(LocalExecutionAttemptConflict):
        await store.prepare_local_execution_attempt(another_authority)
    await asyncio.sleep(1.05)
    cancellation = ordinary_cancellation_reconciliation_request(await store.load_task("b"))
    await store.reconcile_task_cancellation(cancellation)
    assert (await store.load_task("b")).status is TaskStatus.CANCELLED
    assert (
        await app.reconcile_task_group("race")
    ).quiescence.status is TaskGroupQuiescenceStatus.DRAINING
    assert await store.claim_task("finalizer", TaskQuery(type="finalize")) is None
    await store.settle_local_execution_attempt(_settlement(authority, _receipt(start)))
    assert (
        await app.reconcile_task_group("race")
    ).quiescence.status is TaskGroupQuiescenceStatus.QUIESCENT
    assert await store.claim_task("finalizer", TaskQuery(type="finalize")) is not None


@pytest.mark.parametrize("timeout", [True, False, 0, -1, float("inf"), float("nan"), "30"])
async def test_invalid_quiescence_bounds_fail_before_admission(timeout):
    with pytest.raises(ValueError):
        request(timeout=timeout)


@pytest.mark.parametrize(
    "worker_id", ["w" * 256, "w" * 257, "é" * 257], ids=["256", "257", "multibyte"]
)
@pytest.mark.parametrize("loser", [False, True])
async def test_public_group_worker_preserves_worker_identity(store, worker_id, loser):
    from cayu.tasks.worker import run_task_worker

    app = CayuApp(task_store=store, enable_logging=False)
    await app.create_task_group(request())
    entered, finish = asyncio.Event(), asyncio.Event()
    observed = []

    async def handler(_app, task, owner):
        assert owner == worker_id and task.worker_id == worker_id
        observed.append(task)
        entered.set()
        await finish.wait()
        if not loser:
            await store.complete_task(
                task.id, {}, worker_id=owner, lease_expires_at=task.lease_expires_at
            )

    worker = asyncio.create_task(
        run_task_worker(
            app,
            store,
            handler,
            worker_id=worker_id,
            query=TaskQuery(type="b"),
            lease_seconds=3,
            max_tasks=1,
            poll_interval_s=0.01,
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 5)
        snapshot = await app.load_task_group("race")
        execution = next(item for item in snapshot.quiescence.executions if item.task_id == "b")
        assert execution.worker_id == worker_id
        assert execution.settled_at is None
        with pytest.raises(TaskGroupConflict, match="does not match"):
            await store._settle_task_group_execution(
                observed[0].model_copy(update={"worker_id": worker_id[:-1] + "x"})
            )
        assert await app.load_task_group("race") == snapshot
        if loser:
            await store.complete_task("a", {})
            assert await store.claim_task("finalizer", TaskQuery(type="finalize")) is None
        finish.set()
        assert await asyncio.wait_for(worker, 10) == 1
        assert len(observed) == 1
        task = await store.load_task("b")
        assert task.status is (TaskStatus.CANCELLED if loser else TaskStatus.COMPLETED)
        snapshot = await app.reconcile_task_group("race")
        execution = next(item for item in snapshot.quiescence.executions if item.task_id == "b")
        assert execution.worker_id == worker_id
        assert execution.started_at == observed[0].started_at
        assert execution.settled_at is not None
        assert snapshot.quiescence.status is TaskGroupQuiescenceStatus.QUIESCENT
        events = await app.list_task_group_events("race")
        await store._settle_task_group_execution(observed[0])
        assert await app.list_task_group_events("race") == events
        if isinstance(store, SQLiteTaskStore):
            address = store.path
            await store.close()
            reopened = SQLiteTaskStore(address)
        elif isinstance(store, PostgresTaskStore):
            address = _postgres_address(store)
            await store.close()
            reopened = PostgresTaskStore(address)
        else:
            reopened = None
        if reopened is not None:
            try:
                restored = CayuApp(task_store=reopened, enable_logging=False)
                assert await restored.load_task_group("race") == snapshot
                await reopened._settle_task_group_execution(observed[0])
                assert await restored.reconcile_task_group("race") == snapshot
                assert await restored.list_task_group_events("race") == events
            finally:
                await reopened.close()
    finally:
        finish.set()
        if not worker.done():
            worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)


@pytest.mark.parametrize("worker_id", ["", " ", " worker", "worker ", 1, True, b"worker"])
async def test_group_execution_obligation_rejects_invalid_worker_identity(worker_id):
    from datetime import UTC, datetime

    from pydantic import ValidationError

    from cayu.tasks.groups import TaskGroupExecutionObligation

    with pytest.raises(ValidationError):
        TaskGroupExecutionObligation(task_id="a", worker_id=worker_id, started_at=datetime.now(UTC))


@pytest.mark.parametrize("expire", [False, True])
async def test_public_worker_waits_for_real_thread_and_explicit_timeout_resolution(store, expire):
    from cayu.tasks.worker import run_task_worker

    app = CayuApp(task_store=store, enable_logging=False)
    await app.create_task_group(request(timeout=0.02 if expire else 10))
    entered = asyncio.Event()
    finish = threading.Event()
    loop = asyncio.get_running_loop()

    def external_work():
        loop.call_soon_threadsafe(entered.set)
        if not finish.wait(10):
            raise RuntimeError("test thread was not released")

    async def handler(_app, _task, _worker):
        await asyncio.to_thread(external_work)

    worker = asyncio.create_task(
        run_task_worker(
            app,
            store,
            handler,
            worker_id="loser-owner",
            query=TaskQuery(type="b"),
            lease_seconds=3,
            max_tasks=1,
            poll_interval_s=0.01,
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 5)
        await store.complete_task("a", {})
        assert await store.claim_task("finalizer-owner", TaskQuery(type="finalize")) is None
        if expire:
            await asyncio.sleep(0.04)
            timed_out = await app.reconcile_task_group("race")
            assert timed_out.quiescence.status is TaskGroupQuiescenceStatus.ATTENTION_REQUIRED
        finish.set()
        await asyncio.wait_for(worker, 8)
        snapshot = await app.reconcile_task_group("race")
        assert snapshot.quiescence.unsettled_task_ids == ()
        if expire:
            assert snapshot.quiescence.status is TaskGroupQuiescenceStatus.ATTENTION_REQUIRED
            assert await store.claim_task("finalizer-owner", TaskQuery(type="finalize")) is None
            resolution = TaskGroupQuiescenceResolution(
                group_id="race",
                request_sha256=snapshot.receipt.request_sha256,
                expected_sequence=snapshot.last_sequence,
                idempotency_key="resolve-timeout",
            )
            resolved = await app.resolve_task_group_quiescence(resolution)
            assert await app.resolve_task_group_quiescence(resolution) == resolved
        else:
            assert snapshot.quiescence.status is TaskGroupQuiescenceStatus.QUIESCENT
        claimed = await store.claim_task("finalizer-owner", TaskQuery(type="finalize"))
        assert claimed is not None and claimed.id == "finalize"
    finally:
        finish.set()
        if not worker.done():
            await asyncio.wait_for(worker, 8)


async def test_caller_cancellation_does_not_acknowledge_a_running_thread(store):
    from cayu.tasks.worker import run_task_worker

    app = CayuApp(task_store=store, enable_logging=False)
    await app.create_task_group(request(timeout=10))
    entered = asyncio.Event()
    finish = threading.Event()
    loop = asyncio.get_running_loop()

    def external_work():
        loop.call_soon_threadsafe(entered.set)
        if not finish.wait(10):
            raise RuntimeError("test thread was not released")

    async def handler(*args):
        await asyncio.to_thread(external_work)

    worker = asyncio.create_task(
        run_task_worker(
            app,
            store,
            handler,
            worker_id="cancelled-owner",
            query=TaskQuery(type="b"),
            lease_seconds=3,
            max_tasks=1,
            poll_interval_s=0.01,
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 5)
        await store.complete_task("a", {})
        worker.cancel()
        await asyncio.sleep(0.05)
        assert not worker.done()
        assert (await app.load_task_group("race")).quiescence.unsettled_task_ids == ("b",)
        assert await store.claim_task("finalizer", TaskQuery(type="finalize")) is None
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(worker, 8)
        assert worker.cancelled() and worker.cancelling() == 1
        assert (
            await app.reconcile_task_group("race")
        ).quiescence.status is TaskGroupQuiescenceStatus.QUIESCENT
        assert await store.claim_task("finalizer", TaskQuery(type="finalize")) is not None
    finally:
        finish.set()
        if not worker.done():
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait_for(worker, 8)


async def test_terminal_task_does_not_release_a_handler_still_finalizing(store):
    from cayu.tasks.worker import fail_managed_task, run_task_worker

    app = CayuApp(task_store=store, enable_logging=False)
    await app.create_task_group(request(timeout=10))
    published = asyncio.Event()
    finish = asyncio.Event()

    async def handler(_app, task, worker_id):
        await fail_managed_task(store, task, worker_id, {"code": "expected_failure"})
        published.set()
        await finish.wait()

    worker = asyncio.create_task(
        run_task_worker(
            app,
            store,
            handler,
            worker_id="finalizing-owner",
            query=TaskQuery(type="b"),
            lease_seconds=3,
            max_tasks=1,
            poll_interval_s=0.01,
        )
    )
    try:
        await asyncio.wait_for(published.wait(), 5)
        await store.complete_task("a", {})
        assert (await store.load_task("b")).status is TaskStatus.FAILED
        snapshot = await app.load_task_group("race")
        assert snapshot.quiescence.unsettled_task_ids == ("b",)
        assert await store.claim_task("finalizer", TaskQuery(type="finalize")) is None
        finish.set()
        await asyncio.wait_for(worker, 8)
        assert (
            await app.load_task_group("race")
        ).quiescence.status is TaskGroupQuiescenceStatus.QUIESCENT
        assert await store.claim_task("finalizer", TaskQuery(type="finalize")) is not None
    finally:
        finish.set()
        if not worker.done():
            await asyncio.wait_for(worker, 8)


async def test_independent_workers_race_winner_and_late_completion(store):
    from cayu.tasks.base import TaskTerminalizationConflict
    from cayu.tasks.worker import complete_managed_task, run_task_worker

    if isinstance(store, SQLiteTaskStore):
        peer = SQLiteTaskStore(store.path)
    elif isinstance(store, PostgresTaskStore):
        await store.load_task("a")  # Initialize the pool through the public read path.
        address = _postgres_address(store)
        peer = PostgresTaskStore(address)
    else:
        peer = store
    apps = [CayuApp(task_store=value, enable_logging=False) for value in (store, peer)]
    await apps[0].create_task_group(request())
    starts = [asyncio.Event(), asyncio.Event()]
    dispatch = asyncio.Event()
    losing_cleanup, release = asyncio.Event(), asyncio.Event()

    async def handler(app, task, worker):
        starts[0 if task.id == "a" else 1].set()
        await dispatch.wait()
        try:
            terminal = await complete_managed_task(app.task_store, task, worker, {"done": True})
        except (TaskGroupConflict, TaskTerminalizationConflict):
            losing_cleanup.set()
            await release.wait()
        else:
            if terminal.status is TaskStatus.CANCELLED:
                # Ordinary cancellation can win the terminalization election
                # before the group guard sees a proposed success.
                losing_cleanup.set()
                await release.wait()
            else:
                assert terminal.status is TaskStatus.COMPLETED

    workers = [
        asyncio.create_task(
            run_task_worker(
                app,
                app.task_store,
                handler,
                worker_id=f"owner-{index}",
                query=TaskQuery(type=identity),
                lease_seconds=10,
                max_tasks=1,
            )
        )
        for index, (app, identity) in enumerate(zip(apps, ("a", "b"), strict=True))
    ]
    try:
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in starts)), 10)
        dispatch.set()
        await asyncio.wait_for(losing_cleanup.wait(), 10)
        snapshot = await apps[1].load_task_group("race")
        assert len(snapshot.decision.successful_task_ids) == 1
        assert snapshot.quiescence.status is TaskGroupQuiescenceStatus.DRAINING
        assert await peer.claim_task("finalizer", TaskQuery(type="finalize")) is None
        release.set()
        assert await asyncio.wait_for(asyncio.gather(*workers), 15) == [1, 1]
        after = await apps[0].reconcile_task_group("race")
        assert after.decision == snapshot.decision
        assert after.quiescence.status is TaskGroupQuiescenceStatus.QUIESCENT
        assert await apps[1].load_task_group("race") == after
        events = await apps[0].list_task_group_events("race")
        assert sum(event.type is TaskGroupEventType.FINALIZER_RELEASED for event in events) == 1
        await apps[1].reconcile_task_group("race")
        assert await apps[1].list_task_group_events("race") == events
    finally:
        dispatch.set()
        release.set()
        for worker in workers:
            if not worker.done():
                worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        if peer is not store:
            await peer.close()


@pytest.mark.parametrize("with_finalizer", [False, True])
async def test_quorum_barrier_cancels_only_noncontributors(store, with_finalizer):
    app = CayuApp(task_store=store, enable_logging=False)
    creation = TaskGroupCreate(
        group_id="quorum",
        graph=TaskGraphCreate(
            graph_id="quorum-graph",
            nodes=tuple(
                TaskGraphNode(task=TaskCreate(task_id=name, type=name))
                for name in (
                    "a",
                    "b",
                    "c",
                    "d",
                    "unrelated",
                    "finalize",
                )
            ),
        ),
        member_task_ids=("a", "b", "c", "d"),
        policy=TaskGroupPolicy(kind="quorum", k=2),
        quiescence=TaskGroupQuiescencePolicy(timeout_seconds=60),
        finalizer_task_id="finalize" if with_finalizer else None,
    )
    await app.create_task_group(creation)
    await store.complete_task("a", {})
    before = await app.load_task_group("quorum")
    assert before.decision is None
    assert before.quiescence.status is TaskGroupQuiescenceStatus.WAITING_DECISION
    if with_finalizer:
        assert await store.claim_task("publisher", TaskQuery(type="finalize")) is None
    await store.complete_task("b", {})
    result = await app.load_task_group("quorum")
    assert result.decision.successful_task_ids == ("a", "b")
    assert result.quiescence.loser_task_ids == ("c", "d")
    assert result.quiescence.status is TaskGroupQuiescenceStatus.QUIESCENT
    assert result.quiescence.finalizer_status is (
        TaskGroupFinalizerStatus.RELEASED if with_finalizer else TaskGroupFinalizerStatus.ABSENT
    )
    for name in ("c", "d"):
        assert (await store.load_task(name)).status is TaskStatus.CANCELLED
        assert await store.claim_task("late", TaskQuery(type=name)) is None
    assert (await store.load_task("unrelated")).status is TaskStatus.PENDING
    events = await app.list_task_group_events("quorum")
    assert [
        event.task_id for event in events if event.type is TaskGroupEventType.CANCELLATION_REQUESTED
    ] == ["c", "d"]
    await app.reconcile_task_group("quorum")
    assert await app.list_task_group_events("quorum") == events


async def test_finalizer_publication_failure_rolls_back_entire_barrier(store, monkeypatch):
    app = CayuApp(task_store=store, enable_logging=False)
    creation = request()
    await app.create_task_group(creation)
    before = await app.load_task_group("race")
    graph = await app.load_task_graph("race-graph")
    tasks = {name: await store.load_task(name) for name in ("a", "b", "finalize")}
    events = await app.list_task_group_events("race")
    with monkeypatch.context() as patch:
        if isinstance(store, InMemoryTaskStore):
            prepare = store._prepare_task_write

            def fail_finalizer(task, **kwargs):
                if task.id == "finalize" and task.status is TaskStatus.PENDING:
                    raise RuntimeError("finalizer publication boundary")
                return prepare(task, **kwargs)

            patch.setattr(store, "_prepare_task_write", fail_finalizer)
        elif isinstance(store, SQLiteTaskStore):
            from cayu.storage import _sqlite_task_groups as groups

            publish = groups.publish

            def fail_publication(*args, **kwargs):
                publish(*args, **kwargs)
                raise RuntimeError("finalizer publication boundary")

            patch.setattr(groups, "publish", fail_publication)
        else:
            from cayu.storage import _postgres_task_groups as groups

            publish = groups.publish

            async def fail_publication(*args, **kwargs):
                await publish(*args, **kwargs)
                raise RuntimeError("finalizer publication boundary")

            patch.setattr(groups, "publish", fail_publication)
        with pytest.raises(RuntimeError, match="finalizer publication boundary"):
            await store.complete_task("a", {})
    assert await app.load_task_group("race") == before
    assert await app.load_task_graph("race-graph") == graph
    assert {name: await store.load_task(name) for name in tasks} == tasks
    assert await app.list_task_group_events("race") == events
    await store.complete_task("a", {})
    result = await app.reconcile_task_group("race")
    assert result.quiescence.finalizer_status is TaskGroupFinalizerStatus.RELEASED
    events = await app.list_task_group_events("race")
    assert sum(event.type is TaskGroupEventType.FINALIZER_RELEASED for event in events) == 1
