"""A claim batch must recheck candidates changed by earlier group transitions."""

import asyncio
from datetime import UTC, datetime, timedelta

from cayu.storage.tasks_sqlite import SQLiteTaskStore
from cayu.tasks import (
    TaskCreate,
    TaskGraphCreate,
    TaskGraphNode,
    TaskGroupCreate,
    TaskGroupPolicy,
    TaskGroupQuiescencePolicy,
    TaskSchedulePolicy,
    TaskStatus,
)


def test_schedule_expiry_can_cancel_another_candidate_in_the_same_batch(tmp_path):
    async def scenario():
        now = datetime(2026, 1, 1, tzinfo=UTC)
        store = SQLiteTaskStore(
            tmp_path / "expiry.sqlite", clock=lambda: now, ownership_clock=lambda: now
        )
        try:
            await store.create_task_group(
                TaskGroupCreate(
                    group_id="all-members",
                    graph=TaskGraphCreate(
                        graph_id="graph",
                        nodes=tuple(
                            TaskGraphNode(
                                task=TaskCreate(
                                    task_id=identity,
                                    type="work",
                                    available_at=now - timedelta(minutes=2),
                                    schedule_policy=TaskSchedulePolicy(
                                        expires_at=now - timedelta(minutes=1)
                                    ),
                                )
                            )
                            for identity in ("a", "b")
                        ),
                    ),
                    member_task_ids=("a", "b"),
                    policy=TaskGroupPolicy(kind="all"),
                    quiescence=TaskGroupQuiescencePolicy(timeout_seconds=60),
                )
            )
            await store.create_task(TaskCreate(task_id="unrelated", type="work"))
            # Expiring a decides the group and cancels b. The snapshot of
            # candidate IDs still contains b, but it is no longer claimable
            # and must not be expired a second time.
            claimed = await store.claim_task("worker")
            assert claimed is not None and claimed.id == "unrelated"
            assert (await store.load_task("a")).status is TaskStatus.CANCELLED
            assert (await store.load_task("b")).status is TaskStatus.CANCELLED
        finally:
            await store.close()

    asyncio.run(scenario())
