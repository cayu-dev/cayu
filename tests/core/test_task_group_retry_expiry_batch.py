"""Retry expiry must recheck members changed by earlier group transitions."""

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
    TaskRetryPolicy,
    TaskStatus,
)


def test_retry_expiry_can_cancel_another_member_in_the_same_batch(tmp_path):
    async def scenario():
        now = datetime(2026, 9, 1, tzinfo=UTC)
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
                                    retry_policy=TaskRetryPolicy(
                                        max_attempts=2, max_elapsed_seconds=10
                                    ),
                                    metadata={
                                        "execution_profile_fingerprint": "b" * 64,
                                        "effect_fingerprint": "c" * 64,
                                    },
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
            now += timedelta(seconds=11)
            # Expiring a cancels b, which is still in the expiry batch.
            # Maintenance must preserve that cancellation and keep claiming.
            claimed = await store.claim_task("worker")
            assert claimed is not None and claimed.id == "unrelated"
            assert claimed.status is TaskStatus.CLAIMED
            assert (await store.load_task("a")).status is TaskStatus.FAILED
            assert (await store.load_task("b")).status is TaskStatus.CANCELLED
            assert await store.claim_task("another-worker") is None
        finally:
            await store.close()

    asyncio.run(scenario())
