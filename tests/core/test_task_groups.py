"""Public task-group conformance across the native task stores."""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from cayu import (
    CayuApp,
    TaskCreate,
    TaskGraphCreate,
    TaskGraphNode,
    TaskGroupCreate,
    TaskGroupPolicy,
    TaskGroupStatus,
)
from cayu.storage.migrations import SchemaMode
from cayu.storage.postgres import PostgresTaskStore
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks.base import InMemoryTaskStore, TaskStatus
from cayu.tasks.groups import TaskGroupConflict, TaskGroupEventType

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(params=["memory", "sqlite", "postgres"])
async def store(request, tmp_path):
    if request.param == "memory":
        yield InMemoryTaskStore()
        return
    value = (
        SQLiteTaskStore(tmp_path / "groups.sqlite")
        if request.param == "sqlite"
        else PostgresTaskStore(
            request.getfixturevalue("postgres_dsn"), schema_mode=SchemaMode.CREATE
        )
    )
    try:
        yield value
    finally:
        await value.close()


def group_request(policy="all", *, setup=False):
    prefix = uuid4().hex
    members = tuple(prefix + suffix for suffix in ("a", "b", "c"))
    root = prefix + "setup"
    nodes = tuple(
        TaskGraphNode(
            task=TaskCreate(task_id=identity, type=prefix),
            prerequisite_task_ids=(root,) if setup else (),
        )
        for identity in members
    )
    if setup:
        nodes += (TaskGraphNode(task=TaskCreate(task_id=root, type=prefix)),)
    return TaskGroupCreate(
        group_id=prefix,
        graph=TaskGraphCreate(graph_id=prefix + "graph", nodes=nodes),
        member_task_ids=members,
        policy=TaskGroupPolicy(kind=policy, k=2 if policy == "quorum" else None),
    )


@pytest.mark.parametrize("policy,required", [("all", 3), ("first_success", 1), ("quorum", 2)])
async def test_group_success_is_durable_and_does_not_stop_late_members(store, policy, required):
    app = CayuApp(task_store=store, enable_logging=False)
    request = group_request(policy)
    receipt = await app.create_task_group(request)
    for index, identity in enumerate(request.member_task_ids):
        await store.complete_task(identity, {"ok": True})
        snapshot = await require_group(app, request.group_id)
        assert snapshot is not None
        assert snapshot.status is (
            TaskGroupStatus.SUCCEEDED if index + 1 >= required else TaskGroupStatus.PENDING
        )
        if index + 1 == required:
            decision = snapshot.decision
        if index + 1 >= required:
            assert snapshot.decision == decision
    assert await app.create_task_group(request) == receipt
    events = await app.list_task_group_events(request.group_id)
    assert sum(e.type is TaskGroupEventType.MEMBER_TERMINAL for e in events) == 3
    assert sum(e.type is TaskGroupEventType.SUCCEEDED for e in events) == 1
    assert sum(e.type is TaskGroupEventType.POLICY_SATISFIED for e in events) == 1
    assert [e.sequence for e in events] == list(range(1, 7))
    assert await app.list_task_group_events(request.group_id, after_sequence=2) == events[2:]


@pytest.mark.parametrize("policy,failures", [("all", 1), ("first_success", 3), ("quorum", 2)])
async def test_impossible_policy_and_late_completion(store, policy, failures):
    app = CayuApp(task_store=store, enable_logging=False)
    request = group_request(policy)
    await app.create_task_group(request)
    for index, identity in enumerate(request.member_task_ids[:failures]):
        await store.fail_task(identity, {"code": "failed"})
        snapshot = await require_group(app, request.group_id)
        assert snapshot.status is (
            TaskGroupStatus.FAILED if index + 1 == failures else TaskGroupStatus.PENDING
        )
    decision = snapshot.decision
    assert decision is not None
    assert decision.reason == "completion_policy_impossible"
    for identity in request.member_task_ids[failures:]:
        await store.complete_task(identity, {})
    assert (await require_group(app, request.group_id)).decision == decision
    events = await app.list_task_group_events(request.group_id)
    assert sum(e.type is TaskGroupEventType.FAILED for e in events) == 1


@pytest.mark.parametrize("outcome", ["success", "failure"])
async def test_setup_is_not_counted_but_skips_feed_group_policy(store, outcome):
    app = CayuApp(task_store=store, enable_logging=False)
    request = group_request("first_success", setup=True)
    await app.create_task_group(request)
    root = request.group_id + "setup"
    await (store.complete_task(root, {}) if outcome == "success" else store.fail_task(root, {}))
    snapshot = await require_group(app, request.group_id)
    assert snapshot.status is (
        TaskGroupStatus.PENDING if outcome == "success" else TaskGroupStatus.FAILED
    )
    assert all(
        m.status is (TaskStatus.PENDING if outcome == "success" else TaskStatus.DEPENDENCY_SKIPPED)
        for m in snapshot.members
    )
    events = await app.list_task_group_events(request.group_id)
    assert all(e.task_id != root for e in events)


@pytest.mark.parametrize("change", ["members", "policy", "graph"])
async def test_exact_replay_rejects_changed_authority(store, change):
    app = CayuApp(task_store=store, enable_logging=False)
    request = group_request()
    await app.create_task_group(request)
    changed = TaskGroupCreate(
        group_id=request.group_id,
        graph=TaskGraphCreate(graph_id="another", nodes=request.graph.nodes)
        if change == "graph"
        else request.graph,
        member_task_ids=request.member_task_ids[:1]
        if change == "members"
        else request.member_task_ids,
        policy=TaskGroupPolicy(kind="first_success") if change == "policy" else request.policy,
    )
    before = await app.list_task_group_events(request.group_id)
    with pytest.raises(TaskGroupConflict):
        await app.create_task_group(changed)
    assert await app.list_task_group_events(request.group_id) == before


async def test_group_cannot_attach_to_existing_graph(store):
    request = group_request()
    await store.create_task_graph(request.graph)
    with pytest.raises(TaskGroupConflict):
        await store.create_task_group(request)
    assert await store.load_task_group(request.group_id) is None


async def test_concurrent_completions_finalize_once(store):
    request = group_request("quorum")
    await store.create_task_group(request)
    await asyncio.gather(
        *(store.complete_task(identity, {}) for identity in request.member_task_ids)
    )
    snapshot = await require_group(store, request.group_id)
    assert snapshot.status is TaskGroupStatus.SUCCEEDED
    events = await store.list_task_group_events(request.group_id)
    assert sum(e.type is TaskGroupEventType.SUCCEEDED for e in events) == 1


@pytest.mark.parametrize(
    "members,k", [((), 1), (("a", "a"), 1), (("foreign",), 1), (("a",), True), (("a",), 2)]
)
async def test_invalid_membership_and_threshold(members, k):
    with pytest.raises(ValueError):
        TaskGroupCreate(
            group_id="g",
            graph=TaskGraphCreate(
                graph_id="graph", nodes=(TaskGraphNode(task=TaskCreate(task_id="a", type="test")),)
            ),
            member_task_ids=members,
            policy=TaskGroupPolicy(kind="quorum", k=k),
        )


@pytest.mark.parametrize("phase", ["admission", "terminal"])
async def test_publication_failure_rolls_back_tasks_graph_and_group(store, monkeypatch, phase):
    from cayu.storage import _postgres_task_groups, _sqlite_task_groups
    from cayu.tasks import _memory_graphs

    app = CayuApp(task_store=store, enable_logging=False)
    request = group_request("first_success", setup=True)
    if phase == "terminal":
        await app.create_task_group(request)
    before = await app.load_task_group(request.group_id)
    before_graph = await app.load_task_graph(request.graph.graph_id)
    with monkeypatch.context() as patch:
        if isinstance(store, InMemoryTaskStore):
            name = (
                "prepare_group_admission" if phase == "admission" else "plan_group_graph_transition"
            )
            original = getattr(_memory_graphs, name)

            def failing(*args, **kwargs):
                original(*args, **kwargs)
                raise RuntimeError("group publication failed")

            patch.setattr(_memory_graphs, name, failing)
        elif isinstance(store, SQLiteTaskStore):
            original = _sqlite_task_groups.publish

            def failing(*args, **kwargs):
                original(*args, **kwargs)
                raise RuntimeError("group publication failed")

            patch.setattr(_sqlite_task_groups, "publish", failing)
        else:
            original = _postgres_task_groups.publish

            async def failing(*args, **kwargs):
                await original(*args, **kwargs)
                raise RuntimeError("group publication failed")

            patch.setattr(_postgres_task_groups, "publish", failing)
        with pytest.raises(Exception):
            if phase == "admission":
                await app.create_task_group(request)
            else:
                await store.fail_task(request.group_id + "setup", {})
    assert await app.load_task_group(request.group_id) == before
    assert await app.load_task_graph(request.graph.graph_id) == before_graph
    if phase == "admission":
        for node in request.graph.nodes:
            assert await store.load_task(node.task.task_id) is None
        await app.create_task_group(request)
    await store.fail_task(request.group_id + "setup", {})
    assert (await require_group(app, request.group_id)).status is TaskGroupStatus.FAILED


async def test_cancellation_after_admission_commit_replays_exactly(store, monkeypatch):
    app = CayuApp(task_store=store, enable_logging=False)
    request = group_request()
    committed = asyncio.Event()
    original = store.create_task_group

    async def lose_acknowledgement(request):
        receipt = await original(request)
        committed.set()
        await asyncio.Future()
        return receipt

    with monkeypatch.context() as patch:
        patch.setattr(store, "create_task_group", lose_acknowledgement)
        task = asyncio.create_task(app.create_task_group(request))
        await asyncio.wait_for(committed.wait(), 10)
        before = await app.list_task_group_events(request.group_id)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled() and task.cancelling() == 1
    receipt = await app.create_task_group(request)
    assert receipt == (await require_group(app, request.group_id)).receipt
    assert await app.list_task_group_events(request.group_id) == before


async def test_claimed_cancellation_is_not_terminal_until_reconciled(store):
    from tests.core.task_terminalization_conformance import (
        ordinary_cancellation_reconciliation_request,
    )

    from cayu.tasks.base import TaskQuery

    app = CayuApp(task_store=store, enable_logging=False)
    request = group_request("all")
    await app.create_task_group(request)
    claimed = await store.claim_task(
        "group-worker", query=TaskQuery(type=request.group_id), lease_seconds=1
    )
    assert claimed is not None
    requested = await store.cancel_task(claimed.id, {"code": "operator"})
    assert (await require_group(app, request.group_id)).status is TaskGroupStatus.PENDING
    reconciliation = ordinary_cancellation_reconciliation_request(requested)
    await asyncio.sleep(1.05)
    result = await store.reconcile_task_cancellation(reconciliation)
    assert result.task.status is TaskStatus.CANCELLED
    assert (await require_group(app, request.group_id)).status is TaskGroupStatus.FAILED
    before = await app.list_task_group_events(request.group_id)
    assert await store.reconcile_task_cancellation(reconciliation) == result
    assert await app.list_task_group_events(request.group_id) == before


async def test_real_worker_executes_setup_and_selected_members(store):
    from cayu.tasks.base import TaskQuery
    from cayu.tasks.worker import complete_managed_task, run_task_worker

    request = group_request("quorum", setup=True)
    app = CayuApp(task_store=store, enable_logging=False)
    await app.create_task_group(request)
    observed = []

    async def handler(_app, task, worker_id):
        observed.append(task.id)
        await complete_managed_task(store, task, worker_id, {"ok": True})

    count = await asyncio.wait_for(
        run_task_worker(
            app,
            store,
            handler,
            worker_id="group-worker",
            query=TaskQuery(type=request.group_id),
            max_tasks=4,
            poll_interval_s=0.01,
            reclaim=False,
            recover_interrupted_handoffs=False,
        ),
        15,
    )
    assert count == 4
    assert observed[0] == request.group_id + "setup"
    snapshot = await require_group(app, request.group_id)
    assert snapshot.status is TaskGroupStatus.SUCCEEDED
    assert all(member.status is TaskStatus.COMPLETED for member in snapshot.members)


async def test_group_inspection_survives_terminal_member_deletion(store):
    from cayu.tasks.base import TaskSessionClosureClaim
    from cayu.tasks.graphs import TaskGraphConflict

    request = group_request("first_success")
    first = request.member_task_ids[0]
    session = request.group_id + "session"
    for node in request.graph.nodes:
        if node.task.task_id == first:
            node.task.session_id = session
    await store.create_task_group(request)
    await store.complete_task(first, {})
    with pytest.raises(TaskGraphConflict):
        await store.delete_session_tasks(session, task_ids=(first,), policy=None)
    for identity in request.member_task_ids[1:]:
        await store.fail_task(identity, {})
    before = await store.load_task_group(request.group_id)
    await store.claim_session_closure(
        TaskSessionClosureClaim(session_id=session, plan_id="c" * 64, task_ids=(first,))
    )
    await store.delete_session_tasks(session, task_ids=(first,), policy=None)
    assert await store.load_task(first) is None
    assert await store.load_task_group(request.group_id) == before
    assert await store.create_task_group(request) == before.receipt


async def require_group(owner, group_id):
    snapshot = await owner.load_task_group(group_id)
    assert snapshot is not None
    return snapshot
