"""Fresh-process durable task retry-series example.

Usage:
    uv sync --extra dev
    PYTHONPATH=src .venv/bin/python examples/task_retry_worker.py

The parent process creates one SQLite-backed retry series, then starts a fresh
worker process for each attempt. Attempt one explicitly reports a retryable
failure; attempt two succeeds. A third fresh worker proves that the completed
series has no runnable attempt and therefore cannot replay completed work.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import tempfile
from decimal import Decimal
from pathlib import Path

from cayu import (
    CayuApp,
    SQLiteTaskStore,
    Task,
    TaskCreate,
    TaskOrder,
    TaskQuery,
    TaskRetryAttemptDisposition,
    TaskRetryAttemptReport,
    TaskRetryPolicy,
    TaskRetrySeriesDisposition,
    TaskStatus,
    run_task_worker,
)


async def _attempt(_app: CayuApp, task: Task, _worker_id: str) -> TaskRetryAttemptReport:
    series = task.retry_series
    if series is None:
        raise RuntimeError("Worker claimed a task without retry-series authority.")
    if series.attempt == 1:
        return TaskRetryAttemptReport(
            idempotency_key=f"attempt:{series.series_id}:1",
            disposition=TaskRetryAttemptDisposition.RETRYABLE_FAILURE,
            error={"code": "temporary_source_unavailable"},
            token_count=4,
            estimated_cost=Decimal("0.01"),
        )
    return TaskRetryAttemptReport(
        idempotency_key=f"attempt:{series.series_id}:{series.attempt}",
        disposition=TaskRetryAttemptDisposition.SUCCEEDED,
        result={"processed": True},
        token_count=3,
        estimated_cost=Decimal("0.01"),
    )


async def _run_worker(database: Path, worker_id: str) -> int:
    store = SQLiteTaskStore(database)
    app = CayuApp(task_store=store, enable_logging=False)
    stop = asyncio.Event()

    async def stop_when_idle() -> None:
        await asyncio.sleep(0.05)
        stop.set()

    stop_task = asyncio.create_task(stop_when_idle())
    try:
        return await run_task_worker(
            app,
            store,
            _attempt,
            worker_id=worker_id,
            query=TaskQuery(type="document-refresh"),
            poll_interval_s=0.01,
            stop=stop,
            max_tasks=1,
        )
    finally:
        await stop_task
        await store.close()


def _worker_process(database: Path, worker_id: str) -> int:
    completed = subprocess.run(
        [sys.executable, __file__, "--worker", str(database), worker_id],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")},
    )
    return int(completed.stdout.strip())


async def _demonstrate(database: Path) -> None:
    store = SQLiteTaskStore(database)
    first = await store.create_task(
        TaskCreate(
            task_id="document-refresh-attempt-1",
            type="document-refresh",
            input={"document_id": "document-42"},
            retry_policy=TaskRetryPolicy(
                max_attempts=3,
                max_total_tokens=10,
                max_estimated_cost=Decimal("0.05"),
                initial_backoff_seconds=0,
            ),
        )
    )
    await store.close()

    assert _worker_process(database, "worker-attempt-1") == 1
    assert _worker_process(database, "worker-attempt-2") == 1
    assert _worker_process(database, "worker-after-completion") == 0

    inspection_store = SQLiteTaskStore(database)
    attempts = await inspection_store.list_tasks(
        TaskQuery(type="document-refresh", order_by=TaskOrder.CREATED_AT_ASC)
    )
    await inspection_store.close()

    assert len(attempts) == 2
    assert attempts[0].id == first.id
    assert [task.status for task in attempts] == [TaskStatus.FAILED, TaskStatus.COMPLETED]
    assert [task.retry_series.attempt for task in attempts if task.retry_series is not None] == [
        1,
        2,
    ]
    series_ids = {task.retry_series.series_id for task in attempts if task.retry_series is not None}
    causal_budget_ids = {
        task.retry_series.causal_budget_id for task in attempts if task.retry_series is not None
    }
    assert len(series_ids) == 1
    assert causal_budget_ids == series_ids
    final_series = attempts[-1].retry_series
    assert final_series is not None
    assert final_series.disposition is TaskRetrySeriesDisposition.SUCCEEDED
    print(
        "series",
        final_series.series_id,
        "attempts",
        len(attempts),
        "tokens",
        final_series.cumulative_tokens,
        "disposition",
        final_series.disposition,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", nargs=2, metavar=("DATABASE", "WORKER_ID"))
    arguments = parser.parse_args()
    if arguments.worker is not None:
        database, worker_id = arguments.worker
        print(asyncio.run(_run_worker(Path(database), worker_id)))
        return
    with tempfile.TemporaryDirectory(prefix="cayu-task-retry-") as directory:
        asyncio.run(_demonstrate(Path(directory) / "tasks.sqlite"))


if __name__ == "__main__":
    main()
