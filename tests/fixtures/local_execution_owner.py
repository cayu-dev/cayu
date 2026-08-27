"""Disposable Cayu owner process for abrupt-death containment tests."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from cayu import (
    CayuApp,
    LocalExecutionAttemptCoordinator,
    LocalExecutionAttemptLimits,
    LocalExecutionAttemptRequest,
    LocalExecutionEffectPolicy,
    SQLiteTaskStore,
    TaskCreate,
)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("database")
    parser.add_argument("attempt_state")
    parser.add_argument("tree_state")
    parser.add_argument("fixture")
    parser.add_argument("source_root")
    args = parser.parse_args()

    store = SQLiteTaskStore(args.database)
    app = CayuApp(task_store=store, enable_logging=False)
    await store.create_task(TaskCreate(task_id="parent-death-task", type="local-execution"))
    task = await store.claim_task("dead-owner", lease_seconds=1)
    assert task is not None
    coordinator = LocalExecutionAttemptCoordinator(store, state_dir=args.attempt_state)
    request = LocalExecutionAttemptRequest(
        effect_lineage_id="parent-death-effect",
        argv=(sys.executable, args.fixture, "root", args.tree_state),
        cwd=args.source_root,
        env={"PYTHONPATH": str(Path(args.source_root) / "src")},
        effect_policy=LocalExecutionEffectPolicy.LOCAL_ONLY,
        limits=LocalExecutionAttemptLimits(
            term_grace_seconds=1,
            kill_grace_seconds=1,
        ),
    )
    await coordinator.run(
        app=app,
        task=task,
        worker_id="dead-owner",
        request=request,
    )


if __name__ == "__main__":
    asyncio.run(main())
