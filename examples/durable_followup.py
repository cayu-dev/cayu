"""Application-owned one-shot follow-up across separate processes.

Run with PYTHONPATH=src and an installed Cayu environment:
    python examples/durable_followup.py schedule --database followups.sqlite --due <UTC-ISO-time>
    python examples/durable_followup.py work --database followups.sqlite
    python examples/durable_followup.py inspect --database followups.sqlite

The producer exits after durable creation. The worker can be stopped before the
deadline and reconstructed. SQLite, not its timer, decides eligibility. This
example records a follow-up decision without contacting a customer or provider.
Real external effects still require application idempotency/reconciliation.
For another occurrence, supply a new deterministic --task-id and an explicit
--due after the application's chosen completion boundary; this is not cron.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from cayu import CayuApp, Task, TaskCreate, TaskQuery, TaskSchedulePolicy
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks import complete_managed_task, run_task_worker


async def run(command: str, database: Path, task_id: str, due: datetime | None) -> None:
    store = SQLiteTaskStore(database)
    app = CayuApp(task_store=store, enable_logging=False)
    try:
        if command == "schedule":
            if due is None:
                raise ValueError("schedule requires an explicit timezone-aware --due")
            task = await app.create_task(
                TaskCreate(
                    task_id=task_id,
                    type="customer-followup",
                    title="Review pending customer follow-up",
                    input={"action": "review_followup"},
                    available_at=due,
                    schedule_policy=TaskSchedulePolicy(),
                )
            )
            print(json.dumps({"task_id": task.id, "status": task.status}), flush=True)
        elif command == "work":
            # Open the durable store before reporting readiness to the operator.
            await store.load_task(task_id)
            print(json.dumps({"worker_ready": True}), flush=True)

            async def followup(_app: CayuApp, task: Task, worker_id: str) -> None:
                await complete_managed_task(
                    store,
                    task,
                    worker_id,
                    {"decision": "followup_ready", "observed_at": datetime.now(UTC).isoformat()},
                )

            await run_task_worker(
                app,
                store,
                followup,
                worker_id="followup-worker",
                query=TaskQuery(type="customer-followup"),
                max_tasks=1,
                poll_interval_s=10,
                minimum_idle_delay_s=10,
                maximum_idle_delay_s=10,
                idle_jitter_ratio=0,
                reclaim=False,
                recover_interrupted_handoffs=False,
            )
            print(json.dumps({"handled": 1}), flush=True)
        elif command == "probe":
            # Useful to demonstrate that no queue admission happens before due,
            # or that a completed occurrence cannot be claimed a second time.
            claimed = await store.claim_task("probe", TaskQuery(type="customer-followup"))
            print(json.dumps({"claimed": None if claimed is None else claimed.id}), flush=True)
            if claimed is not None:
                assert claimed.lease_expires_at is not None
                await store.release_task(
                    claimed.id, "probe", lease_expires_at=claimed.lease_expires_at
                )
        else:
            task = await store.load_task(task_id)
            events = await app.list_task_schedule_events(task_id)
            print(
                json.dumps(
                    {
                        "task_id": task_id,
                        "status": None if task is None else task.status,
                        "result": None if task is None else task.result,
                        "events": [event.type.value for event in events],
                    }
                ),
                flush=True,
            )
    finally:
        await store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("schedule", "work", "probe", "inspect"))
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--task-id", default="customer-42-followup-1")
    parser.add_argument("--due", type=datetime.fromisoformat)
    args = parser.parse_args()
    asyncio.run(run(args.command, args.database, args.task_id, args.due))


if __name__ == "__main__":
    main()
