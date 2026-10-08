from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from cayu.sessions.access import (
    SessionAccessDenied,
    SessionAccessRule,
    SessionAccessScope,
    SessionAccessSelector,
)
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks.access import ScopedTaskAccess
from cayu.tasks.base import InMemoryTaskStore
from cayu.tasks.creation import TaskCreate
from cayu.tasks.queries import TaskQuery


async def conformance(store):
    prefix = uuid4().hex

    def scope(org):
        rule = SessionAccessRule(
            selectors=(SessionAccessSelector(key="organization", values=(org,)),)
        )
        return SessionAccessScope(read=(rule,), create=(rule,), modify=(rule,))

    current = scope("acme")

    async def resolve():
        return current

    acme = ScopedTaskAccess(store, admitted=current, resolve=resolve)

    async def other_resolve():
        return scope("other")

    other = ScopedTaskAccess(store, admitted=scope("other"), resolve=other_resolve)
    own = await acme.create(
        TaskCreate(task_id=prefix + "a", type="test", title="same"), labels={"organization": "acme"}
    )
    foreign = await other.create(
        TaskCreate(task_id=prefix + "b", type="test", title="same"),
        labels={"organization": "other"},
    )
    await acme.create(
        TaskCreate(task_id=prefix + "c", type="test", title="same"), labels={"organization": "acme"}
    )
    assert (await acme.load(own.id)).id == own.id
    for task_id in (foreign.id, prefix + "missing"):
        with pytest.raises(SessionAccessDenied):
            await acme.load(task_id)
    for offset in (0, 1):
        page = await acme.list(TaskQuery(q=prefix, limit=1, offset=offset))
        assert len(page) == 1 and page[0].id != foreign.id
    assert not await acme.list(TaskQuery(q=prefix, limit=1, offset=2))
    for operation in (acme.pause, acme.resume, acme.cancel):
        with pytest.raises(SessionAccessDenied):
            await operation(foreign.id)
        with pytest.raises(SessionAccessDenied):
            await operation(prefix + "missing")
    assert (await acme.pause(own.id)).status == "paused"
    assert (await acme.resume(own.id)).status == "pending"
    assert (await acme.cancel(own.id)).status == "cancelled"
    assert (await other.load(foreign.id)).status == "pending"
    with pytest.raises(SessionAccessDenied):
        await acme.create(
            TaskCreate(task_id=prefix + "denied", type="test", title="denied"),
            labels={"organization": "other"},
        )
    assert await store.load_task(prefix + "denied") is None
    with pytest.raises(SessionAccessDenied):
        await acme.create(
            TaskCreate(
                task_id=prefix + "child", parent_task_id=foreign.id, type="test", title="denied"
            ),
            labels={"organization": "acme"},
        )
    assert await store.load_task(prefix + "child") is None
    from cayu.tasks.graphs import TaskGraphCreate, TaskGraphNode
    from cayu.tasks.groups import TaskGroupCreate, TaskGroupPolicy

    def graph(name):
        return TaskGraphCreate(
            graph_id=prefix + name,
            nodes=(TaskGraphNode(task=TaskCreate(task_id=prefix + name + "member", type="test")),),
        )

    own_graph = graph("own-graph")
    foreign_graph = graph("foreign-graph")
    await acme.create_graph(own_graph, labels={"organization": "acme"})
    await other.create_graph(foreign_graph, labels={"organization": "other"})
    assert (await acme.graph(own_graph.graph_id)).receipt.access_invocation.access_labels == {
        "organization": "acme"
    }
    assert await acme.graph_events(own_graph.graph_id)
    for operation in (acme.graph, acme.graph_events):
        with pytest.raises(SessionAccessDenied):
            await operation(foreign_graph.graph_id)
        with pytest.raises(SessionAccessDenied):
            await operation(prefix + "missing-graph")
    with pytest.raises(SessionAccessDenied):
        await acme.create_graph(foreign_graph, labels={"organization": "acme"})
    own_group_graph = graph("own-group-graph")
    foreign_group_graph = graph("foreign-group-graph")
    for target, specification, organization in (
        (acme, own_group_graph, "acme"),
        (other, foreign_group_graph, "other"),
    ):
        await target.create_group(
            TaskGroupCreate(
                group_id=specification.graph_id + "group",
                graph=specification,
                member_task_ids=(specification.nodes[0].task.task_id,),
                policy=TaskGroupPolicy(kind="all"),
            ),
            labels={"organization": organization},
        )
    assert await acme.group(own_group_graph.graph_id + "group")
    assert await acme.group_events(own_group_graph.graph_id + "group")
    for operation in (acme.group, acme.group_events):
        with pytest.raises(SessionAccessDenied):
            await operation(foreign_group_graph.graph_id + "group")
        with pytest.raises(SessionAccessDenied):
            await operation(prefix + "missing-group")
    current = SessionAccessScope(read=(SessionAccessRule(allow_all=True),))
    with pytest.raises(SessionAccessDenied):
        await acme.load(foreign.id)
    current = SessionAccessScope()
    assert not await acme.list()
    with pytest.raises(SessionAccessDenied):
        await acme.load(own.id)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_task_access(backend, tmp_path):
    async def run():
        store = (
            InMemoryTaskStore() if backend == "memory" else SQLiteTaskStore(tmp_path / "tasks.db")
        )
        try:
            await conformance(store)
        finally:
            if backend != "memory":
                await store.close()

    asyncio.run(run())


def test_postgres_task_access(postgres_dsn):
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.postgres import PostgresTaskStore

    async def run():
        store = PostgresTaskStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            await conformance(store)
        finally:
            await store.close()

    asyncio.run(run())
