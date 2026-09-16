"""Boundary and independent-owner tests for group admission and settlement."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from tests.core.test_task_groups import anyio_backend as anyio_backend
from tests.core.test_task_groups import group_request, require_group
from tests.core.test_task_groups import store as store

from cayu import (
    CayuApp,
    TaskCreate,
    TaskGraphCreate,
    TaskGraphNode,
    TaskGroupCreate,
    TaskGroupPolicy,
    TaskGroupStatus,
)
from cayu._validation import canonical_durable_json_bytes
from cayu.storage.migrations import SchemaMode
from cayu.storage.postgres import PostgresTaskStore
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks.base import InMemoryTaskStore, TaskQuery, TaskStatus
from cayu.tasks.graphs import TASK_GRAPH_MAX_BYTES
from cayu.tasks.groups import TaskGroupConflict, TaskGroupEventType, task_group_request_sha256
from cayu.tasks.scheduling import TaskSchedulePolicy

pytestmark = pytest.mark.anyio


def claim_group_request(scheduled):
    request = group_request()
    if scheduled:
        for node in request.graph.nodes:
            node.task.available_at = datetime.now(UTC) - timedelta(seconds=1)
            node.task.schedule_policy = TaskSchedulePolicy()
    return request


@pytest.mark.parametrize("scheduled", [False, True], ids=["unscheduled", "scheduled"])
async def test_group_claim_inspection_matches_task_state(store, scheduled):
    app = CayuApp(task_store=store, enable_logging=False)
    request = claim_group_request(scheduled)
    await app.create_task_group(request)
    before = await app.list_task_group_events(request.group_id)
    claimed = await store.claim_task("claim-worker", query=TaskQuery(type=request.group_id))
    assert claimed is not None and claimed.status is TaskStatus.CLAIMED
    task = await store.load_task(claimed.id)
    assert task is not None
    assert task == claimed
    snapshot = await require_group(app, request.group_id)
    assert next(m.status for m in snapshot.members if m.task_id == task.id) is task.status
    assert snapshot.status is TaskGroupStatus.PENDING
    assert await app.list_task_group_events(request.group_id) == before
    assert claimed.lease_expires_at is not None
    released = await store.release_task(
        claimed.id, "claim-worker", lease_expires_at=claimed.lease_expires_at
    )
    assert released.status is TaskStatus.PENDING
    snapshot = await require_group(app, request.group_id)
    assert next(m.status for m in snapshot.members if m.task_id == released.id) is released.status
    assert await app.list_task_group_events(request.group_id) == before


@pytest.mark.parametrize("scheduled", [False, True], ids=["unscheduled", "scheduled"])
async def test_sqlite_group_claim_inspection_survives_reopen(tmp_path, scheduled):
    path = tmp_path / "claim-group.sqlite"
    request = claim_group_request(scheduled)
    store = SQLiteTaskStore(path)
    try:
        app = CayuApp(task_store=store, enable_logging=False)
        receipt = await app.create_task_group(request)
        claimed = await store.claim_task("claim-worker", query=TaskQuery(type=request.group_id))
        assert claimed is not None and claimed.status is TaskStatus.CLAIMED
        events = await app.list_task_group_events(request.group_id)
    finally:
        await store.close()
    reopened = SQLiteTaskStore(path)
    try:
        app = CayuApp(task_store=reopened, enable_logging=False)
        task = await reopened.load_task(claimed.id)
        assert task == claimed
        snapshot = await require_group(app, request.group_id)
        assert (
            next(m.status for m in snapshot.members if m.task_id == claimed.id)
            is TaskStatus.CLAIMED
        )
        assert snapshot.decision is None
        assert await app.create_task_group(request) == receipt
        assert await app.load_task_group(request.group_id) == snapshot
        assert await app.list_task_group_events(request.group_id) == events
    finally:
        await reopened.close()


@pytest.mark.parametrize("scheduled", [False, True], ids=["unscheduled", "scheduled"])
async def test_sqlite_group_claim_publication_failure_rolls_back(tmp_path, monkeypatch, scheduled):
    from cayu.storage import _sqlite_task_groups

    store = SQLiteTaskStore(tmp_path / "claim-rollback.sqlite")
    try:
        app = CayuApp(task_store=store, enable_logging=False)
        request = claim_group_request(scheduled)
        await app.create_task_group(request)
        snapshot = await require_group(app, request.group_id)
        tasks = [await store.load_task(identity) for identity in request.member_task_ids]
        graph_events = await app.list_task_graph_events(request.graph.graph_id)
        group_events = await app.list_task_group_events(request.group_id)
        original = _sqlite_task_groups.publish
        attempted = []

        def fail_after_publication(store, publication, *, creating=False):
            original(store, publication, creating=creating)
            attempted.append(publication.snapshot)
            raise RuntimeError("claim group publication failed")

        with monkeypatch.context() as patch:
            patch.setattr(_sqlite_task_groups, "publish", fail_after_publication)
            with pytest.raises(RuntimeError, match="claim group publication failed"):
                await store.claim_task("claim-worker", query=TaskQuery(type=request.group_id))
        assert len(attempted) == 1
        assert any(m.status is TaskStatus.CLAIMED for m in attempted[0].members)
        assert [await store.load_task(identity) for identity in request.member_task_ids] == tasks
        assert await app.load_task_group(request.group_id) == snapshot
        assert await app.list_task_graph_events(request.graph.graph_id) == graph_events
        assert await app.list_task_group_events(request.group_id) == group_events
        claimed = await store.claim_task("retry-worker", query=TaskQuery(type=request.group_id))
        assert claimed is not None
        snapshot = await require_group(app, request.group_id)
        assert (
            next(m.status for m in snapshot.members if m.task_id == claimed.id)
            is TaskStatus.CLAIMED
        )
    finally:
        await store.close()


@pytest.mark.parametrize("policy", ["all", "first_success", "quorum"])
async def test_singleton_group(store, policy):
    identity = uuid4().hex
    request = TaskGroupCreate(
        group_id=identity,
        graph=TaskGraphCreate(
            graph_id=identity,
            nodes=(TaskGraphNode(task=TaskCreate(task_id=identity, type=identity)),),
        ),
        member_task_ids=(identity,),
        policy=TaskGroupPolicy(kind=policy, k=1 if policy == "quorum" else None),
    )
    app = CayuApp(task_store=store, enable_logging=False)
    await app.create_task_group(request)
    await store.complete_task(identity, {})
    assert (await require_group(app, identity)).status is TaskGroupStatus.SUCCEEDED


async def test_maximum_selected_members_skip_atomically(store):
    identity = uuid4().hex
    members = tuple(identity + f"-{i:03}" for i in range(127))
    root = identity + "-root"
    request = TaskGroupCreate(
        group_id=identity,
        graph=TaskGraphCreate(
            graph_id=identity,
            nodes=(
                TaskGraphNode(task=TaskCreate(task_id=root, type=identity)),
                *(
                    TaskGraphNode(
                        task=TaskCreate(task_id=m, type=identity), prerequisite_task_ids=(root,)
                    )
                    for m in members
                ),
            ),
        ),
        member_task_ids=members,
        policy=TaskGroupPolicy(kind="quorum", k=64),
    )
    app = CayuApp(task_store=store, enable_logging=False)
    await app.create_task_group(request)
    await store.fail_task(root, {})
    snapshot = await require_group(app, identity)
    assert snapshot.status is TaskGroupStatus.FAILED
    events = await app.list_task_group_events(identity, limit=1000)
    assert len(events) == 130
    assert sum(e.type is TaskGroupEventType.MEMBER_TERMINAL for e in events) == 127


async def test_group_request_checks_combined_bytes_not_individual_graph_only():
    # Fill the graph near its public bound with individually bounded task inputs.
    nodes = tuple(
        TaskGraphNode(
            task=TaskCreate(task_id=f"m-{i:03}", type="test", input={"blob": "x" * 30000})
        )
        for i in range(32)
    )
    graph = TaskGraphCreate(graph_id="g", nodes=nodes)
    size = len(canonical_durable_json_bytes(graph.model_dump(mode="json"), "graph"))
    nodes[0].task.input["blob"] += "x" * (TASK_GRAPH_MAX_BYTES - size - 8)
    graph = TaskGraphCreate(graph_id="g", nodes=nodes)
    with pytest.raises(ValueError, match="canonical byte bound"):
        TaskGroupCreate(
            group_id="group",
            graph=graph,
            member_task_ids=tuple(f"m-{i:03}" for i in range(32)),
            policy=TaskGroupPolicy(kind="all"),
        )


async def test_canonical_reordering_replays_without_new_events(store):
    app = CayuApp(task_store=store, enable_logging=False)
    request = group_request("quorum")
    receipt = await app.create_task_group(request)
    reordered = TaskGroupCreate(
        group_id=request.group_id,
        graph=TaskGraphCreate(
            graph_id=request.graph.graph_id, nodes=tuple(reversed(request.graph.nodes))
        ),
        member_task_ids=tuple(reversed(request.member_task_ids)),
        policy=request.policy,
    )
    assert task_group_request_sha256(request) == task_group_request_sha256(reordered)
    assert await app.create_task_group(reordered) == receipt
    assert len(await app.list_task_group_events(request.group_id)) == 1


async def test_unsupported_store_fails_before_publication():
    class Unsupported(InMemoryTaskStore):
        supports_task_groups = False

        async def create_task_group(self, request):
            pytest.fail("Unsupported store must not be invoked")

    store = Unsupported()
    app = CayuApp(task_store=store, enable_logging=False)
    request = group_request()
    with pytest.raises(NotImplementedError):
        await app.create_task_group(request)
    assert await store.load_task_graph(request.graph.graph_id) is None


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
@pytest.mark.parametrize("race", ["admission", "outcome"])
async def test_independent_store_instances_share_one_decision(backend, race, tmp_path, request):
    address = (
        str(tmp_path / "concurrent.sqlite")
        if backend == "sqlite"
        else request.getfixturevalue("postgres_dsn")
    )
    cls = SQLiteTaskStore if backend == "sqlite" else PostgresTaskStore
    first, second = (
        cls(address, schema_mode=SchemaMode.CREATE),
        cls(address, schema_mode=SchemaMode.CREATE),
    )
    group = group_request("first_success")
    try:
        apps = [CayuApp(task_store=value, enable_logging=False) for value in (first, second)]
        if race == "admission":
            receipts = await asyncio.wait_for(
                asyncio.gather(*(app.create_task_group(group) for app in apps)), 20
            )
            assert receipts[0] == receipts[1]
        else:
            await apps[0].create_task_group(group)
        await asyncio.wait_for(
            asyncio.gather(
                first.complete_task(group.member_task_ids[0], {}),
                second.fail_task(group.member_task_ids[1], {}),
            ),
            20,
        )
        snapshot = await require_group(apps[1], group.group_id)
        assert snapshot.status is TaskGroupStatus.SUCCEEDED
        assert snapshot.decision.successful_task_ids == (group.member_task_ids[0],)
        before = await apps[0].list_task_group_events(group.group_id)
        assert sum(e.type is TaskGroupEventType.SUCCEEDED for e in before) == 1
        assert await apps[1].create_task_group(group) == snapshot.receipt
        assert await apps[1].list_task_group_events(group.group_id) == before
    finally:
        await first.close()
        await second.close()
    reopened = cls(address)
    try:
        assert await reopened.load_task_group(group.group_id) == snapshot
        assert await reopened.list_task_group_events(group.group_id) == before
    finally:
        await reopened.close()


async def test_group_identity_conflict_does_not_create_new_graph(store):
    request = group_request()
    app = CayuApp(task_store=store, enable_logging=False)
    await app.create_task_group(request)
    other = group_request()
    collision = TaskGroupCreate(
        group_id=request.group_id,
        graph=other.graph,
        member_task_ids=other.member_task_ids,
        policy=other.policy,
    )
    with pytest.raises(TaskGroupConflict):
        await app.create_task_group(collision)
    assert await app.load_task_graph(other.graph.graph_id) is None


async def test_public_group_rejects_existing_graph_with_typed_conflict(store):
    request = group_request()
    app = CayuApp(task_store=store, enable_logging=False)
    await app.create_task_graph(request.graph)
    with pytest.raises(TaskGroupConflict):
        await app.create_task_group(request)
    assert await app.load_task_group(request.group_id) is None


@pytest.mark.parametrize("kind", ["conflict", "unavailable"])
async def test_store_group_errors_remain_typed_and_secret_safe(kind):
    from cayu.tasks.groups import TaskGroupUnavailable
    from cayu.vaults.redaction import SecretRedactor

    expected = TaskGroupConflict if kind == "conflict" else TaskGroupUnavailable
    secret = "group-store-error-canary"

    class Failing(InMemoryTaskStore):
        async def load_task_group(self, group_id):
            raise expected(secret)

    app = CayuApp(
        task_store=Failing(), secret_redactor=SecretRedactor(secret), enable_logging=False
    )
    with pytest.raises(expected) as raised:
        await app.load_task_group("safe")
    assert secret not in str(raised.value) + repr(raised.value)
    assert raised.value.__context__ is None
