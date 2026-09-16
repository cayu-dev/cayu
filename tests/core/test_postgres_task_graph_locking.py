"""Graph and ordinary task writers share closure exclusion, not each other's locks."""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from cayu.storage.migrations import SchemaMode
from cayu.storage.postgres import PostgresTaskStore
from cayu.tasks.base import TaskCreate, TaskSessionClosureClaim, TaskStatus
from cayu.tasks.graphs import TaskGraphCreate, TaskGraphEventType, TaskGraphNode


@pytest.mark.parametrize("outcome", ["complete", "fail"])
def test_cross_graph_outcomes_share_session_closure_locks(postgres_dsn, outcome):
    async def run():
        prefix = uuid4().hex
        sources = {prefix + "-a", prefix + "-c"}
        entered = set()
        both_updated = asyncio.Event()

        class BarrierStore(PostgresTaskStore):
            async def _record_task_transition(self, cur, prior, current, **kwargs):
                if current.id in sources:
                    entered.add(current.id)
                    if entered == sources:
                        both_updated.set()
                    await asyncio.wait_for(both_updated.wait(), 10)
                await super()._record_task_transition(cur, prior, current, **kwargs)

        left = BarrierStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        right = BarrierStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        operations = []
        try:
            await left._ensure_ready()
            await right._ensure_ready()
            for root, child, first, second in (("a", "b", "x", "y"), ("c", "d", "y", "x")):
                await left.create_task_graph(
                    TaskGraphCreate(
                        graph_id=prefix + root,
                        nodes=(
                            TaskGraphNode(
                                task=TaskCreate(
                                    task_id=prefix + "-" + root,
                                    type="test",
                                    session_id=prefix + first,
                                )
                            ),
                            TaskGraphNode(
                                task=TaskCreate(
                                    task_id=prefix + "-" + child,
                                    type="test",
                                    session_id=prefix + second,
                                ),
                                prerequisite_task_ids=(prefix + "-" + root,),
                            ),
                        ),
                    )
                )
            for store, root in ((left, "a"), (right, "c")):
                operation = store.complete_task if outcome == "complete" else store.fail_task
                operations.append(asyncio.create_task(operation(prefix + "-" + root, {})))
            await asyncio.wait_for(asyncio.gather(*operations), 15)
            for root, child in (("a", "b"), ("c", "d")):
                snapshot = await right.load_task_graph(prefix + root)
                assert snapshot is not None
                members = {member.task_id: member for member in snapshot.members}
                assert members[prefix + "-" + root].status is (
                    TaskStatus.COMPLETED if outcome == "complete" else TaskStatus.FAILED
                )
                assert members[prefix + "-" + child].status is (
                    TaskStatus.PENDING if outcome == "complete" else TaskStatus.DEPENDENCY_SKIPPED
                )
                events = await right.list_task_graph_events(prefix + root)
                assert len(events) == 5
                assert sum(event.type is TaskGraphEventType.TERMINAL for event in events) == 1
                assert (
                    sum(
                        event.task_id == prefix + "-" + child
                        and event.type
                        is (
                            TaskGraphEventType.READY
                            if outcome == "complete"
                            else TaskGraphEventType.SKIPPED
                        )
                        for event in events
                    )
                    == 1
                )
        finally:
            for operation in operations:
                if not operation.done():
                    operation.cancel()
            await asyncio.gather(*operations, return_exceptions=True)
            await left.close()
            await right.close()

    asyncio.run(run())


def test_graph_admission_and_ordinary_contracted_creation_share_lock_order(postgres_dsn):
    from tests.core.test_verified_work_contracts import _contract

    async def run():
        prefix = uuid4().hex
        ordinary_has_authority = asyncio.Event()
        graph_requests_authority = asyncio.Event()

        class OrdinaryStore(PostgresTaskStore):
            async def _ensure_session_authority(self, cur, session_id, authority_kind):
                await super()._ensure_session_authority(cur, session_id, authority_kind)
                ordinary_has_authority.set()
                await asyncio.wait_for(graph_requests_authority.wait(), 10)

        class GraphStore(PostgresTaskStore):
            async def _ensure_session_authority(self, cur, session_id, authority_kind):
                graph_requests_authority.set()
                await super()._ensure_session_authority(cur, session_id, authority_kind)

        ordinary = OrdinaryStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        graphs = GraphStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        operations = []
        try:
            await ordinary._ensure_ready()
            await graphs._ensure_ready()
            contract = await ordinary.publish_work_contract(_contract(contract_id=prefix))
            operations.append(
                asyncio.create_task(
                    ordinary.create_task(
                        TaskCreate(
                            task_id=prefix + "-ordinary",
                            type="test",
                            session_id=prefix,
                            work_contract=contract.reference(),
                        )
                    )
                )
            )
            await asyncio.wait_for(ordinary_has_authority.wait(), 10)
            request = TaskGraphCreate(
                graph_id=prefix,
                nodes=(
                    TaskGraphNode(
                        task=TaskCreate(
                            task_id=prefix + "-graph",
                            type="test",
                            session_id=prefix,
                            work_contract=contract.reference(),
                        )
                    ),
                ),
            )
            operations.append(asyncio.create_task(graphs.create_task_graph(request)))
            _, receipt = await asyncio.wait_for(asyncio.gather(*operations), 15)
            assert await graphs.load_task(prefix + "-ordinary") is not None
            assert await graphs.load_task(prefix + "-graph") is not None
            assert await graphs.create_task_graph(request) == receipt
            assert len(await graphs.list_task_graph_events(prefix)) == 2
        finally:
            for operation in operations:
                if not operation.done():
                    operation.cancel()
            await asyncio.gather(*operations, return_exceptions=True)
            await ordinary.close()
            await graphs.close()

    asyncio.run(run())


def test_exclusive_closure_still_waits_for_shared_task_writer(postgres_dsn):
    async def run():
        prefix = uuid4().hex
        inserted = asyncio.Event()
        release = asyncio.Event()

        class PausedWriter(PostgresTaskStore):
            async def _record_task_transition(self, cur, prior, current, **kwargs):
                await super()._record_task_transition(cur, prior, current, **kwargs)
                inserted.set()
                await release.wait()

        writer = PausedWriter(postgres_dsn, schema_mode=SchemaMode.CREATE)
        closer = PostgresTaskStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        operations = []
        try:
            await writer._ensure_ready()
            await closer._ensure_ready()
            operations.append(
                asyncio.create_task(
                    writer.create_task(
                        TaskCreate(
                            task_id=prefix,
                            type="test",
                            session_id=prefix,
                        )
                    )
                )
            )
            await asyncio.wait_for(inserted.wait(), 10)
            operations.append(
                asyncio.create_task(
                    closer.claim_session_closure(
                        TaskSessionClosureClaim(session_id=prefix, plan_id="a" * 64, task_ids=())
                    )
                )
            )
            async with closer._connection() as conn, conn.cursor() as cur:
                async with asyncio.timeout(10):
                    while True:
                        await cur.execute(
                            "SELECT 1 FROM pg_locks WHERE locktype = 'advisory' "
                            "AND NOT granted AND mode = 'ExclusiveLock' "
                            "AND classid::bigint = ((hashtextextended(%s, 0) >> 32) & 4294967295) "
                            "AND objid::bigint = (hashtextextended(%s, 0) & 4294967295)",
                            ("cayu-task-session-closure:" + prefix,) * 2,
                        )
                        if await cur.fetchone() is not None:
                            break
                        await asyncio.sleep(0.01)
            assert not operations[1].done()
            release.set()
            await asyncio.wait_for(operations[0], 10)
            with pytest.raises(ValueError, match="closure set changed"):
                await asyncio.wait_for(operations[1], 10)
            assert await closer.load_session_closure_claim(prefix) is None
            assert await closer.load_task(prefix) is not None
        finally:
            release.set()
            for operation in operations:
                if not operation.done():
                    operation.cancel()
            await asyncio.gather(*operations, return_exceptions=True)
            await writer.close()
            await closer.close()

    asyncio.run(run())
