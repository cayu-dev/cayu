"""Persistent graph transactions through the real PostgreSQL task store."""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from cayu.storage.migrations import SchemaMode
from cayu.storage.postgres import PostgresTaskStore
from cayu.tasks.base import TaskCreate, TaskStatus
from cayu.tasks.graphs import TaskGraphCreate, TaskGraphEventType, TaskGraphNode


def test_postgres_graph_outcomes_reconstruct(postgres_dsn: str) -> None:
    async def run() -> None:
        request = TaskGraphCreate(
            graph_id="outcomes",
            nodes=(
                TaskGraphNode(
                    task=TaskCreate(task_id="join", type="test"), prerequisite_task_ids=("a", "b")
                ),
                TaskGraphNode(task=TaskCreate(task_id="b", type="test")),
                TaskGraphNode(task=TaskCreate(task_id="a", type="test")),
                TaskGraphNode(
                    task=TaskCreate(task_id="tail", type="test"), prerequisite_task_ids=("join",)
                ),
            ),
        )
        first = PostgresTaskStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            receipt = await first.create_task_graph(request)
            initial = await first.load_task_graph("outcomes")
            assert initial is not None
            assert {
                member.task_id
                for member in initial.members
                if member.status is TaskStatus.WAITING_DEPENDENCIES
            } == {"join", "tail"}
            await first.complete_task("a", {})
        finally:
            await first.close()
        second = PostgresTaskStore(postgres_dsn)
        try:
            assert await second.create_task_graph(request) == receipt
            await second.complete_task("b", {})
            joined = await second.load_task("join")
            assert joined is not None and joined.status is TaskStatus.PENDING
            await second.fail_task("join", {"code": "test"})
            final = await second.load_task_graph("outcomes")
            assert final is not None
            assert final.members[-1].status is TaskStatus.DEPENDENCY_SKIPPED
            events = await second.list_task_graph_events("outcomes")
            assert (
                len(
                    [
                        event
                        for event in events
                        if event.task_id == "join" and event.type is TaskGraphEventType.READY
                    ]
                )
                == 1
            )
            assert len([event for event in events if event.type is TaskGraphEventType.SKIPPED]) == 1
        finally:
            await second.close()

    asyncio.run(run())


def test_postgres_graph_lock_wait_does_not_consume_claim_lease(
    postgres_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cayu.storage import _postgres_task_graphs as graphs

    async def run() -> None:
        store = PostgresTaskStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        waiting = asyncio.Event()
        original_lock = graphs.lock_graph

        async def observed_lock(cur, graph_id):
            waiting.set()
            await original_lock(cur, graph_id)

        claim = None
        try:
            await store.create_task_graph(
                TaskGraphCreate(
                    graph_id="lease-wait",
                    nodes=(
                        TaskGraphNode(
                            task=TaskCreate(task_id="lease-wait-task", type="lease-wait")
                        ),
                    ),
                )
            )
            async with store._connection() as conn, conn.cursor() as cur:
                await original_lock(cur, "lease-wait")
                monkeypatch.setattr(graphs, "lock_graph", observed_lock)
                from cayu.tasks.base import TaskQuery

                claim = asyncio.create_task(
                    store.claim_task("lease-worker", TaskQuery(type="lease-wait"), lease_seconds=1)
                )
                await asyncio.wait_for(waiting.wait(), 5)
                # The graph lock remains owned longer than the requested lease.
                await asyncio.sleep(1.1)
                assert not claim.done()
                await conn.commit()
            task = await asyncio.wait_for(claim, 5)
            assert task is not None
            assert task.lease_expires_at == task.updated_at + timedelta(seconds=1)
            # Check against the same physical authority as the actual store.
            async with store._connection() as conn, conn.cursor() as cur:
                now = await store._database_now(cur)
            assert task.lease_expires_at > now
        finally:
            if claim is not None and not claim.done():
                claim.cancel()
                await asyncio.gather(claim, return_exceptions=True)
            await store.close()

    asyncio.run(run())


def test_postgres_concurrent_prerequisites_elect_one_join(postgres_dsn: str) -> None:
    async def run() -> None:
        left = PostgresTaskStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        right = PostgresTaskStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        request = TaskGraphCreate(
            graph_id="concurrent",
            nodes=(
                TaskGraphNode(task=TaskCreate(task_id="race-a", type="test")),
                TaskGraphNode(task=TaskCreate(task_id="race-b", type="test")),
                TaskGraphNode(
                    task=TaskCreate(task_id="race-join", type="test"),
                    prerequisite_task_ids=("race-a", "race-b"),
                ),
            ),
        )
        try:
            await left.create_task_graph(request)
            await asyncio.wait_for(
                asyncio.gather(
                    left.complete_task("race-a", {}),
                    right.complete_task("race-b", {}),
                ),
                10,
            )
            claims = await asyncio.wait_for(
                asyncio.gather(left.claim_task("left"), right.claim_task("right")), 10
            )
            assert [task.id for task in claims if task is not None] == ["race-join"]
            events = await right.list_task_graph_events("concurrent")
            releases = [
                event
                for event in events
                if event.type is TaskGraphEventType.READY and event.task_id == "race-join"
            ]
            assert len(releases) == 1
        finally:
            await left.close()
            await right.close()

    asyncio.run(run())


def test_postgres_graph_cancel_before_commit_rolls_back_all_members(
    postgres_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cayu import CayuApp
    from cayu.storage import _postgres_task_graphs as graphs

    async def run() -> None:
        store = PostgresTaskStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        app = CayuApp(task_store=store, enable_logging=False)
        entered = asyncio.Event()
        release = asyncio.Event()
        original = graphs.insert_events

        async def pause_before_events(cur, events):
            entered.set()
            await release.wait()
            await original(cur, events)

        request = TaskGraphCreate(
            graph_id="cancel-before-commit",
            nodes=(
                TaskGraphNode(task=TaskCreate(task_id="cancel-root", type="test")),
                TaskGraphNode(
                    task=TaskCreate(task_id="cancel-child", type="test"),
                    prerequisite_task_ids=("cancel-root",),
                ),
            ),
        )
        caller = None
        try:
            with monkeypatch.context() as patch:
                patch.setattr(graphs, "insert_events", pause_before_events)
                caller = asyncio.create_task(app.create_task_graph(request))
                await asyncio.wait_for(entered.wait(), 10)
                caller.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(caller, 10)
                assert caller.cancelled() and caller.cancelling() == 1
            assert await store.load_task("cancel-root") is None
            assert await store.load_task("cancel-child") is None
            assert await store.load_task_graph(request.graph_id) is None
            await app.create_task_graph(request)
            assert len(await app.list_task_graph_events(request.graph_id)) == 3
        finally:
            release.set()
            if caller is not None:
                if not caller.done():
                    caller.cancel()
                await asyncio.gather(caller, return_exceptions=True)
            await store.close()

    asyncio.run(run())


def test_postgres_graph_creation_commit_then_raise_replays(postgres_dsn: str) -> None:
    class LostAcknowledgementStore(PostgresTaskStore):
        async def _run_verified_work_mutation(self, operation):
            await super()._run_verified_work_mutation(operation)
            raise RuntimeError("acknowledgement lost after graph commit")

    async def run() -> None:
        request = TaskGraphCreate(
            graph_id="ack",
            nodes=(
                TaskGraphNode(task=TaskCreate(task_id="ack-root", type="test")),
                TaskGraphNode(
                    task=TaskCreate(task_id="ack-child", type="test"),
                    prerequisite_task_ids=("ack-root",),
                ),
            ),
        )
        first = LostAcknowledgementStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            with pytest.raises(RuntimeError, match="acknowledgement lost"):
                await first.create_task_graph(request)
        finally:
            await first.close()
        replay = PostgresTaskStore(postgres_dsn)
        try:
            before = await replay.list_task_graph_events("ack")
            receipt = await replay.create_task_graph(request)
            assert receipt.task_ids == ("ack-child", "ack-root")
            assert await replay.list_task_graph_events("ack") == before
            await replay.complete_task("ack-root", {})
            child = await replay.load_task("ack-child")
            assert child is not None and child.status is TaskStatus.PENDING
        finally:
            await replay.close()

    asyncio.run(run())
