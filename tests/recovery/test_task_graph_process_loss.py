"""Recover graph publication after real owner death at lost acknowledgement."""

from __future__ import annotations

import asyncio
import sys
from uuid import uuid4

import pytest

from cayu.storage.postgres import PostgresTaskStore
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks.base import TaskStatus
from cayu.tasks.graphs import TaskGraphEventType

pytestmark = pytest.mark.process

_CHILD = r"""
import asyncio
import sys
from cayu import CayuApp
from cayu.storage.migrations import SchemaMode
from cayu.storage.postgres import PostgresTaskStore
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks.base import TaskCreate
from cayu.tasks.graphs import TaskGraphCreate, TaskGraphNode

async def main():
    backend, address, graph_id, outcome = sys.argv[1:]
    base = SQLiteTaskStore if backend == 'sqlite' else PostgresTaskStore

    class LostAcknowledgementStore(base):
        async def complete_task(self, *args, **kwargs):
            result = await super().complete_task(*args, **kwargs)
            print('committed', flush=True)
            await asyncio.Future()
            return result

        async def fail_task(self, *args, **kwargs):
            result = await super().fail_task(*args, **kwargs)
            print('committed', flush=True)
            await asyncio.Future()
            return result

    store = LostAcknowledgementStore(address, schema_mode=SchemaMode.CREATE)
    app = CayuApp(task_store=store, enable_logging=False)
    await app.create_task_graph(TaskGraphCreate(graph_id=graph_id, nodes=(
        TaskGraphNode(task=TaskCreate(task_id=graph_id+'-child', type=graph_id),
                      prerequisite_task_ids=(graph_id+'-root',)),
        TaskGraphNode(task=TaskCreate(task_id=graph_id+'-root', type=graph_id)),
    )))
    operation = store.complete_task if outcome == 'complete' else store.fail_task
    await operation(graph_id+'-root', {'code': 'test'})

asyncio.run(main())
"""


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
@pytest.mark.parametrize("outcome", ["complete", "fail"])
def test_graph_outcome_survives_sigkill_before_acknowledgement(
    backend,
    outcome,
    tmp_path,
    request,
) -> None:
    address = (
        str(tmp_path / "graph.sqlite")
        if backend == "sqlite"
        else request.getfixturevalue("postgres_dsn")
    )
    graph_id = "process-graph-" + uuid4().hex

    async def run():
        child = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            _CHILD,
            backend,
            address,
            graph_id,
            outcome,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            assert child.stdout is not None
            marker = await asyncio.wait_for(child.stdout.readline(), 30)
            assert marker == b"committed\n"
            child.kill()
            await asyncio.wait_for(child.wait(), 10)
            assert child.returncode is not None and child.returncode < 0
        finally:
            if child.returncode is None:
                child.kill()
                await asyncio.wait_for(child.wait(), 10)
        store = SQLiteTaskStore(address) if backend == "sqlite" else PostgresTaskStore(address)
        try:
            snapshot = await store.load_task_graph(graph_id)
            assert snapshot is not None
            members = {member.task_id: member for member in snapshot.members}
            expected = (
                TaskStatus.PENDING if outcome == "complete" else TaskStatus.DEPENDENCY_SKIPPED
            )
            assert members[graph_id + "-child"].status is expected
            events = await store.list_task_graph_events(graph_id)
            kind = TaskGraphEventType.READY if outcome == "complete" else TaskGraphEventType.SKIPPED
            assert (
                len(
                    [
                        event
                        for event in events
                        if event.task_id == graph_id + "-child" and event.type is kind
                    ]
                )
                == 1
            )
            claims = await asyncio.gather(
                store.claim_task("replacement-a"), store.claim_task("replacement-b")
            )
            assert [task.id for task in claims if task is not None] == (
                [graph_id + "-child"] if outcome == "complete" else []
            )
        finally:
            await store.close()

    asyncio.run(run())
